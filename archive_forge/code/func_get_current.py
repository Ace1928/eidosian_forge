import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import ray
from ray._private.ray_constants import env_integer
from ray.util.annotations import DeveloperAPI
from ray.util.scheduling_strategies import SchedulingStrategyT
@staticmethod
def get_current() -> 'DataContext':
    """Get or create a singleton context.

        If the context has not yet been created in this process, it will be
        initialized with default settings.
        """
    global _default_context
    with _context_lock:
        if _default_context is None:
            _default_context = DataContext(target_max_block_size=DEFAULT_TARGET_MAX_BLOCK_SIZE, target_shuffle_max_block_size=DEFAULT_SHUFFLE_TARGET_MAX_BLOCK_SIZE, target_min_block_size=DEFAULT_TARGET_MIN_BLOCK_SIZE, streaming_read_buffer_size=DEFAULT_STREAMING_READ_BUFFER_SIZE, enable_pandas_block=DEFAULT_ENABLE_PANDAS_BLOCK, optimize_fuse_stages=DEFAULT_OPTIMIZE_FUSE_STAGES, optimize_fuse_read_stages=DEFAULT_OPTIMIZE_FUSE_READ_STAGES, optimize_fuse_shuffle_stages=DEFAULT_OPTIMIZE_FUSE_SHUFFLE_STAGES, optimize_reorder_stages=DEFAULT_OPTIMIZE_REORDER_STAGES, actor_prefetcher_enabled=DEFAULT_ACTOR_PREFETCHER_ENABLED, use_push_based_shuffle=DEFAULT_USE_PUSH_BASED_SHUFFLE, pipeline_push_based_shuffle_reduce_tasks=True, scheduling_strategy=DEFAULT_SCHEDULING_STRATEGY, scheduling_strategy_large_args=DEFAULT_SCHEDULING_STRATEGY_LARGE_ARGS, large_args_threshold=DEFAULT_LARGE_ARGS_THRESHOLD, use_polars=DEFAULT_USE_POLARS, new_execution_backend=DEFAULT_NEW_EXECUTION_BACKEND, use_streaming_executor=DEFAULT_USE_STREAMING_EXECUTOR, eager_free=DEFAULT_EAGER_FREE, decoding_size_estimation=DEFAULT_DECODING_SIZE_ESTIMATION_ENABLED, min_parallelism=DEFAULT_MIN_PARALLELISM, enable_tensor_extension_casting=DEFAULT_ENABLE_TENSOR_EXTENSION_CASTING, enable_auto_log_stats=DEFAULT_AUTO_LOG_STATS, trace_allocations=DEFAULT_TRACE_ALLOCATIONS, optimizer_enabled=DEFAULT_OPTIMIZER_ENABLED, execution_options=ray.data.ExecutionOptions(), use_ray_tqdm=DEFAULT_USE_RAY_TQDM, enable_progress_bars=DEFAULT_ENABLE_PROGRESS_BARS, enable_get_object_locations_for_metrics=DEFAULT_ENABLE_GET_OBJECT_LOCATIONS_FOR_METRICS, use_runtime_metrics_scheduling=DEFAULT_USE_RUNTIME_METRICS_SCHEDULING, write_file_retry_on_errors=DEFAULT_WRITE_FILE_RETRY_ON_ERRORS)
        return _default_context