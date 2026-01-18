from collections import defaultdict, deque
from functools import partial
import statistics
from typing import ClassVar, Deque, Dict, Optional
import torch
@classmethod
def get_common_timings(cls, event_recorders: Dict[str, Deque['CudaEventRecorder']], description: str) -> str:
    all_timings_str = f'{description}:\n'
    for event_name, event_recorder_list in event_recorders.items():
        time_taken_list = [event_recorder.find_time_elapsed() for event_recorder in event_recorder_list]
        all_timings_str += '{}: Time taken: avg: {}, std: {}, count: {}\n'.format(event_name, statistics.mean(time_taken_list), statistics.pstdev(time_taken_list), len(time_taken_list))
    return all_timings_str