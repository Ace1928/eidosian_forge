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
def summarize_worker_startup_time():
    workers = list_workers(detail=True, filters=[('worker_type', '=', 'WORKER')], limit=10000, raise_on_missing_output=False)
    time_to_launch = []
    time_to_initialize = []
    for worker in workers:
        launch_time = worker.get('worker_launch_time_ms')
        launched_time = worker.get('worker_launched_time_ms')
        start_time = worker.get('start_time_ms')
        if launched_time > 0:
            time_to_launch.append(launched_time - launch_time)
        if start_time:
            time_to_initialize.append(start_time - launched_time)
    time_to_launch.sort()
    time_to_initialize.sort()

    def print_latencies(latencies):
        print(f'Avg: {round(sum(latencies) / len(latencies), 2)} ms')
        print(f'P25: {round(latencies[int(len(latencies) * 0.25)], 2)} ms')
        print(f'P50: {round(latencies[int(len(latencies) * 0.5)], 2)} ms')
        print(f'P95: {round(latencies[int(len(latencies) * 0.95)], 2)} ms')
        print(f'P99: {round(latencies[int(len(latencies) * 0.99)], 2)} ms')
    print('Time to launch workers')
    print_latencies(time_to_launch)
    print('=======================')
    print('Time to initialize workers')
    print_latencies(time_to_initialize)