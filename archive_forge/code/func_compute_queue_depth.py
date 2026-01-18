import functools
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
from torch.autograd import _KinetoEvent
from torch.autograd.profiler import profile
from torch.profiler import DeviceType
def compute_queue_depth(self):
    """
        Computes queue_depth at each event. This will calculate the queue depth data for
        All the events in the tree.
        This will return a list of Interval of queue depth data of cuda launch and kernels.
        """
    assert self.profile.kineto_results is not None
    cuda_event_list = self.profile.kineto_results.events()

    def is_cuda_launch_kernel(e):
        return e.name == 'cudaLaunchKernel'

    def is_cuda_kernel(e):
        return e.device_type() == DeviceType.CUDA and 'mem' not in e.name.lower()
    cuda_launch_events = sorted((e for e in cuda_event_list if is_cuda_launch_kernel(e)), key=lambda x: x.start_us())
    cuda_kernel_events = sorted((e for e in cuda_event_list if is_cuda_kernel(e)), key=lambda x: x.start_us())
    self.cuda_events = sorted(cuda_launch_events + cuda_kernel_events, key=lambda x: x.start_us())
    kernel_mapping: Dict[_KinetoEvent, int] = {}
    last_mapped_kernel = 0
    for cuda_launch_event in cuda_launch_events:
        index = index_of_first_match(cuda_kernel_events, lambda x: x.linked_correlation_id() == cuda_launch_event.linked_correlation_id(), start=last_mapped_kernel)
        kernel_mapping[cuda_launch_event] = index
        last_mapped_kernel = index if index is not None else last_mapped_kernel
    current_kernel_index = 0
    spawned_kernel_index = -1
    all_events = cuda_launch_events + cuda_kernel_events + self.events

    def new_old_event_comparator(event):
        if hasattr(event, 'start_us'):
            return event.start_us() * 1000
        if hasattr(event, 'start_time_ns'):
            return event.start_time_ns
        raise Exception('Unknown Event Type')
    queue_depth_list: List[Interval] = []
    all_events.sort(key=new_old_event_comparator)
    for event in all_events:
        if hasattr(event, 'start_us'):
            start_time = event.start_us() * 1000
            end_time = (event.start_us() + event.duration_us()) * 1000
            if event in kernel_mapping and kernel_mapping[event] is not None:
                spawned_kernel_index = kernel_mapping[event]
        elif hasattr(event, 'start_time_ns'):
            start_time = event.start_time_ns
            end_time = event.end_time_ns
        while current_kernel_index < len(cuda_kernel_events) and cuda_kernel_events[current_kernel_index].start_us() * 1000 <= start_time:
            current_kernel_index += 1
        current_queue_depth = spawned_kernel_index - current_kernel_index + 1
        current_queue_depth = max(current_queue_depth, 0)
        if hasattr(event, 'start_us'):
            queue_depth_list.append(Interval(start_time, end_time, current_queue_depth))
        elif hasattr(event, 'start_time_ns'):
            self.metrics[EventKey(event)].queue_depth = current_queue_depth
    return queue_depth_list