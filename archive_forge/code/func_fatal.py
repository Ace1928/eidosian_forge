import logging as _logging
import os as _os
import sys as _sys
import _thread
import time as _time
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['logging.fatal'])
def fatal(msg, *args, **kwargs):
    get_logger().fatal(msg, *args, **kwargs)