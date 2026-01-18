from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.buckets.anywhere_caches import pause_anywhere_cache_task
def get_task_iterator(self, args, task_status_queue):
    progress_callbacks.workload_estimator_callback(task_status_queue, len(args.id))
    for id_str in args.id:
        bucket_name, _, anywhere_cache_id = id_str.rpartition('/')
        yield pause_anywhere_cache_task.PauseAnywhereCacheTask(bucket_name, anywhere_cache_id)