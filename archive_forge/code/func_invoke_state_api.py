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
def invoke_state_api(verify_cb: Callable, state_api_fn: Callable, state_stats: StateAPIStats=GLOBAL_STATE_STATS, key_suffix: Optional[str]=None, print_result: Optional[bool]=False, err_msg: Optional[str]=None, **kwargs):
    """Invoke a State API

    Args:
        - verify_cb: Callback that takes in the response from `state_api_fn` and
            returns a boolean, indicating the correctness of the results.
        - state_api_fn: Function of the state API
        - state_stats: Stats
        - kwargs: Keyword arguments to be forwarded to the `state_api_fn`
    """
    if 'timeout' not in kwargs:
        kwargs['timeout'] = STATE_LIST_TIMEOUT
    kwargs['raise_on_missing_output'] = False
    res = None
    try:
        state_stats.total_calls += 1
        state_stats.pending_calls += 1
        t_start = time.perf_counter()
        res = state_api_fn(**kwargs)
        t_end = time.perf_counter()
        if print_result:
            pprint.pprint(res)
        metric = StateAPIMetric(t_end - t_start, len(res))
        if key_suffix:
            key = f'{state_api_fn.__name__}_{key_suffix}'
        else:
            key = state_api_fn.__name__
        state_stats.calls[key].append(metric)
        assert verify_cb(res), f'Calling State API failed. len(res)=({len(res)}): {err_msg}'
    except Exception as e:
        traceback.print_exc()
        assert False, f'Calling {state_api_fn.__name__}({kwargs}) failed with {repr(e)}.'
    finally:
        state_stats.pending_calls -= 1
    return res