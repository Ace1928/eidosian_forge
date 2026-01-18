import asyncio
import sys
from copy import deepcopy
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass, field
import logging
import numpy as np
import pprint
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple, Union
from ray.util.state import list_tasks
import ray
from ray.actor import ActorHandle
from ray.util.state import list_workers
from ray._private.gcs_utils import GcsAioClient, GcsChannel
from ray.util.state.state_manager import StateDataSourceClient
from ray.dashboard.state_aggregator import (
def aggregate_perf_results(state_stats: StateAPIStats=GLOBAL_STATE_STATS):
    """Aggregate stats of state API calls

    Return:
        This returns a dict of below fields:
            - max_{api_key_name}_latency_sec:
                Max latency of call to {api_key_name}
            - {api_key_name}_result_size_with_max_latency:
                The size of the result (or the number of bytes for get_log API)
                for the max latency invocation
            - avg/p99/p95/p50_{api_key_name}_latency_sec:
                The percentile latency stats
            - avg_state_api_latency_sec:
                The average latency of all the state apis tracked
    """
    state_stats = deepcopy(state_stats)
    perf_result = {}
    for api_key_name, metrics in state_stats.calls.items():
        latency_key = f'max_{api_key_name}_latency_sec'
        size_key = f'{api_key_name}_result_size_with_max_latency'
        metric = max(metrics, key=lambda metric: metric.latency_sec)
        perf_result[latency_key] = metric.latency_sec
        perf_result[size_key] = metric.result_size
        latency_list = np.array([metric.latency_sec for metric in metrics])
        key = f'avg_{api_key_name}_latency_sec'
        perf_result[key] = np.average(latency_list)
        key = f'p99_{api_key_name}_latency_sec'
        perf_result[key] = np.percentile(latency_list, 99)
        key = f'p95_{api_key_name}_latency_sec'
        perf_result[key] = np.percentile(latency_list, 95)
        key = f'p50_{api_key_name}_latency_sec'
        perf_result[key] = np.percentile(latency_list, 50)
    all_state_api_latency = sum((metric.latency_sec for metric_samples in state_stats.calls.values() for metric in metric_samples))
    perf_result['avg_state_api_latency_sec'] = all_state_api_latency / state_stats.total_calls if state_stats.total_calls != 0 else -1
    return perf_result