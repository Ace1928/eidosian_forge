import asyncio
from collections import deque, defaultdict
import dataclasses
from dataclasses import field
import logging
from typing import List, Dict, Optional, Set, Deque, Callable
import ray
from ray.workflow.common import (
from ray.workflow.workflow_context import WorkflowTaskContext
def construct_scheduling_plan(self, task_id: TaskID) -> None:
    """Analyze upstream dependencies of a task to construct the scheduling plan."""
    if self.get_input(task_id) is not None:
        return
    visited_nodes = set()
    dag_visit_queue = deque([task_id])
    while dag_visit_queue:
        tid = dag_visit_queue.popleft()
        if tid in visited_nodes:
            continue
        visited_nodes.add(tid)
        self.pending_input_set[tid] = set()
        for in_task_id in self.upstream_dependencies[tid]:
            self.reference_set[in_task_id].add(tid)
            task_input = self.get_input(in_task_id)
            if task_input is None:
                self.pending_input_set[tid].add(in_task_id)
                dag_visit_queue.append(in_task_id)
        if tid in self.latest_continuation:
            if self.pending_input_set[tid]:
                raise ValueError('A task that already returns a continuation cannot be pending.')
            self.construct_scheduling_plan(self.latest_continuation[tid])
        elif not self.pending_input_set[tid]:
            self.append_frontier_to_run(tid)