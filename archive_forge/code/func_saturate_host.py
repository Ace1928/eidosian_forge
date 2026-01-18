import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def saturate_host(self) -> None:
    """Saturate host by assigning replicates to unused devices with enough memory.
        It uses a greedy approach to find a next available set of devices to place all split
        partitions: For each used device, it searches for an idle device with minimal memory
        size that can hold all the partition located on that device; If the search is successful
        for all used devices, it then assigns the new devices' logical ID to the corresponding
        partition.
        """
    device_to_partitions, device_to_left_mem_bytes, no_device_partitions = get_device_partition_stats(self.partitions, self.devices)
    assert len(no_device_partitions) == 0, f'Expect no_device_partitions has 0 device, but get {len(no_device_partitions)}'
    used_devices = [d for d in self.devices if len(device_to_partitions[d]) > 0]
    replicated_device_to_used_device: Dict[Device, Device] = {}
    while len(used_devices) * 2 + len(replicated_device_to_used_device) <= len(self.devices):
        success = True
        idle_devices = [d for d in self.devices if d not in used_devices and d not in replicated_device_to_used_device]
        temp_replicate_mapping = {}
        for used_device in used_devices:
            available_devices = [d for d in idle_devices if d.available_mem_bytes >= used_device.available_mem_bytes - device_to_left_mem_bytes[used_device]]
            if len(available_devices) == 0:
                success = False
                break
            new_device = min(available_devices, key=lambda d: d.available_mem_bytes)
            idle_devices.remove(new_device)
            temp_replicate_mapping[new_device] = used_device
        if not success:
            break
        replicated_device_to_used_device.update(temp_replicate_mapping)
    for replicate_device, original_device in replicated_device_to_used_device.items():
        logical_id = replicate_device.logical_id
        for partition in device_to_partitions[original_device]:
            partition.logical_device_ids.append(logical_id)
    for p in self.partitions:
        print(p.logical_device_ids)