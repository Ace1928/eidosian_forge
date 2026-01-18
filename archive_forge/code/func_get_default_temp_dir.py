import os
from .start_hook_base import RayOnSparkStartHook
from .utils import get_spark_session
import logging
import threading
import time
def get_default_temp_dir(self):
    return _DATABRICKS_DEFAULT_TMP_DIR