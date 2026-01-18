import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, List
from wandb.vendor.pynvml import pynvml
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
def gpu_in_use_by_this_process(gpu_handle: 'GPUHandle', pid: int) -> bool:
    if psutil is None:
        return False
    try:
        base_process = psutil.Process(pid=pid)
    except psutil.NoSuchProcess:
        return False
    our_processes = base_process.children(recursive=True)
    our_processes.append(base_process)
    our_pids = {process.pid for process in our_processes}
    compute_pids = {process.pid for process in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)}
    graphics_pids = {process.pid for process in pynvml.nvmlDeviceGetGraphicsRunningProcesses(gpu_handle)}
    pids_using_device = compute_pids | graphics_pids
    return len(pids_using_device & our_pids) > 0