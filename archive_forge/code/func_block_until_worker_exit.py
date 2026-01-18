import threading
from absl import logging
from tensorflow.python.distribute.failure_handling.failure_handling_util import detect_platform
from tensorflow.python.distribute.failure_handling.failure_handling_util import PlatformDevice
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework.errors import AbortedError
from tensorflow.python.framework.errors import CancelledError
from tensorflow.python.framework.errors import InternalError
from tensorflow.python.framework.errors import UnavailableError
from tensorflow.python.util.tf_export import tf_export
def block_until_worker_exit(self):
    """Block coordinator until workers exit.

    In some rare cases, another error could be raised during the
    preemption grace period. This will cause the coordinator to reconnect to the
    same TPU workers, which will be killed later. It prevents the coordinator to
    reconnect to new TPU workers, and falls back to a hard restart. To avoid
    this situation, this method will block the coordinator to reconnect until
    workers exit. This method will be a no-op for non-TPU platform.
    """
    if self._platform != PlatformDevice.INTERNAL_TPU:
        return
    try:
        context.context().get_config_key_value('BLOCK_TILL_EXIT')
    except InternalError as e:
        if 'Coordination service is not enabled.' not in e.message:
            raise
        logging.info('Workers exited.')
    except (AbortedError, CancelledError, UnavailableError):
        logging.info('Workers exited.')