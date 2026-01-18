import enum
import os
import sys
import requests
from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
def default_tpu_exit_fn():
    """Default exit function to run after saving checkpoint for TPUStrategy.

  For TPUStrategy, we want the coordinator to exit after workers are down so
  that restarted coordinator would not connect to workers scheduled to be
  preempted. This function achieves so by attempting to get a key-value store
  from coordination service, which will block until workers are done and then
  returns with error. Then we have the coordinator sys.exit(42) to re-schedule
  the job.
  """
    logging.info('Waiting for workers to exit...')
    try:
        context.context().get_config_key_value('BLOCK_TILL_EXIT')
    except:
        logging.info('Restarting cluster due to preemption.')
        sys.exit(42)