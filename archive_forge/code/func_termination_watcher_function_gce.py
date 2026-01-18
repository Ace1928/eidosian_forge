import enum
import os
import sys
import requests
from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
def termination_watcher_function_gce():
    result = request_compute_metadata('instance/maintenance-event') == 'TERMINATE_ON_HOST_MAINTENANCE'
    return result