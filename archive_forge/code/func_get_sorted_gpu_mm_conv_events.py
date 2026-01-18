import contextlib
import time
import os
import json
import torch
from torch.profiler import profile, ProfilerActivity
def get_sorted_gpu_mm_conv_events(events):

    def is_mm_conv_event(event):
        return 'name' in event and ('gemm' in event['name'] or 'conv' in event['name'] or 'cutlass' in event['name'] or ('wgrad' in event['name']))
    gpu_events = get_sorted_gpu_events(events)
    sorted_events = []
    for event in gpu_events:
        if not is_mm_conv_event(event):
            continue
        sorted_events.append(event)
    return sorted_events