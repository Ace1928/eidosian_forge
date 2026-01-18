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
def get_temp_filename(self, filename):
    suffix = os.extsep + random_file_extension()
    path = os.path.dirname(filename)
    name = os.path.basename(filename)
    temp_filename = name[:self._MAX_FILENAME_LEN - len(suffix)] + suffix
    return os.path.join(path, temp_filename)