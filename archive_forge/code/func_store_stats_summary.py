import ray
import ray._private.profiling as profiling
import ray._private.services as services
import ray._private.utils as utils
import ray._private.worker
from ray._private import ray_constants
from ray._private.state import GlobalState
from ray._raylet import GcsClientOptions
def store_stats_summary(reply):
    """Returns formatted string describing object store stats in all nodes."""
    store_summary = '--- Aggregate object store stats across all nodes ---\n'
    store_summary += 'Plasma memory usage {} MiB, {} objects, {}% full, {}% needed\n'.format(int(reply.store_stats.object_store_bytes_used / (1024 * 1024)), reply.store_stats.num_local_objects, round(100 * reply.store_stats.object_store_bytes_used / reply.store_stats.object_store_bytes_avail, 2), round(100 * reply.store_stats.object_store_bytes_primary_copy / reply.store_stats.object_store_bytes_avail, 2))
    if reply.store_stats.object_store_bytes_fallback > 0:
        store_summary += 'Plasma filesystem mmap usage: {} MiB\n'.format(int(reply.store_stats.object_store_bytes_fallback / (1024 * 1024)))
    if reply.store_stats.spill_time_total_s > 0:
        store_summary += 'Spilled {} MiB, {} objects, avg write throughput {} MiB/s\n'.format(int(reply.store_stats.spilled_bytes_total / (1024 * 1024)), reply.store_stats.spilled_objects_total, int(reply.store_stats.spilled_bytes_total / (1024 * 1024) / reply.store_stats.spill_time_total_s))
    if reply.store_stats.restore_time_total_s > 0:
        store_summary += 'Restored {} MiB, {} objects, avg read throughput {} MiB/s\n'.format(int(reply.store_stats.restored_bytes_total / (1024 * 1024)), reply.store_stats.restored_objects_total, int(reply.store_stats.restored_bytes_total / (1024 * 1024) / reply.store_stats.restore_time_total_s))
    if reply.store_stats.consumed_bytes > 0:
        store_summary += 'Objects consumed by Ray tasks: {} MiB.\n'.format(int(reply.store_stats.consumed_bytes / (1024 * 1024)))
    if reply.store_stats.object_pulls_queued:
        store_summary += 'Object fetches queued, waiting for available memory.'
    return store_summary