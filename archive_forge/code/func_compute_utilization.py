import contextlib
import time
import os
import json
import torch
from torch.profiler import profile, ProfilerActivity
def compute_utilization(filename: str, total_length: float):
    """
    Process the chrome traces outputs by the pytorch profiler to compute GPU Utilization
    and percent of times spent on matmul and convolution

    Args:
        filename(str): Name of chrome traces file produced by pytorch profiler

        total_length(float): total length of the process without profiler in second

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)
    """
    events = get_chrome_trace_events(filename)
    global gpu_pids
    gpu_pids = []
    for event in events:
        if 'name' not in event:
            continue
        if event['name'] == 'process_labels' and 'GPU' in event['args']['labels']:
            gpu_pids.append(event['pid'])
    total_length = total_length * 1000000.0
    sorted_gpu_events = get_sorted_gpu_events(events)
    utilization = get_duration(sorted_gpu_events) / total_length
    sorted_gpu_mm_conv_events = get_sorted_gpu_mm_conv_events(events)
    mm_conv_utilization = get_duration(sorted_gpu_mm_conv_events) / total_length
    return (utilization, mm_conv_utilization)