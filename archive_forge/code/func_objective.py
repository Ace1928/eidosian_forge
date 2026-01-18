import contextlib
import math
import os
import warnings
from cupy._core import _optimize_config
from cupyx import profiler
def objective(trial):
    args = suggest_func(trial)
    max_total_time = optimize_config.max_total_time_per_trial
    try:
        perf = profiler.benchmark(target_func, args, max_duration=max_total_time)
        return perf.gpu_times.mean()
    except Exception as e:
        if isinstance(e, ignore_error):
            return math.inf
        else:
            raise e