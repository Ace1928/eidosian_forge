import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def construct_memory_table(workers_stats: List, group_by: GroupByType=GroupByType.NODE_ADDRESS, sort_by=SortingType.OBJECT_SIZE) -> MemoryTable:
    memory_table_entries = []
    for core_worker_stats in workers_stats:
        pid = core_worker_stats['pid']
        is_driver = core_worker_stats.get('workerType') == 'DRIVER'
        node_address = core_worker_stats['ipAddress']
        object_refs = core_worker_stats.get('objectRefs', [])
        for object_ref in object_refs:
            memory_table_entry = MemoryTableEntry(object_ref=object_ref, node_address=node_address, is_driver=is_driver, pid=pid)
            if memory_table_entry.is_valid():
                memory_table_entries.append(memory_table_entry)
    memory_table = MemoryTable(memory_table_entries, group_by_type=group_by, sort_by_type=sort_by)
    return memory_table