import random
import time
import functools
import math
import os
import socket
import stat
import string
import logging
import threading
import io
from collections import defaultdict
from botocore.exceptions import IncompleteReadError
from botocore.exceptions import ReadTimeoutError
from s3transfer.compat import SOCKET_ERROR
from s3transfer.compat import rename_file
from s3transfer.compat import seekable
from s3transfer.compat import fallocate
def decrement(self):
    """Decrement the count by one"""
    with self._lock:
        if self._count == 0:
            raise RuntimeError('Counter is at zero. It cannot dip below zero')
        self._count -= 1
        if self._is_finalized and self._count == 0:
            self._callback()