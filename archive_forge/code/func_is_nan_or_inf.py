import os
import socket
from contextlib import closing
import logging
import queue
import threading
from typing import Optional
import numpy as np
from ray.air.constants import _ERROR_REPORT_TIMEOUT
def is_nan_or_inf(value):
    return is_nan(value) or np.isinf(value)