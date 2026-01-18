import threading
import time
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
def start_server(cluster_resolver, protocol):
    """Start a server and block the process from exiting."""
    if not (cluster_resolver.task_type == 'worker' or cluster_resolver.task_type == 'ps'):
        raise ValueError('Unexpected task_type to start a server: {}'.format(cluster_resolver.task_type))
    server = server_lib.Server(cluster_resolver.cluster_spec().as_cluster_def(), job_name=cluster_resolver.task_type, task_index=cluster_resolver.task_id, protocol=protocol)
    logging.info('TensorFlow server started for job %s, task %d.', cluster_resolver.task_type, cluster_resolver.task_id)
    server.join()