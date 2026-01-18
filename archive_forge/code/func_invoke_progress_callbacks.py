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
def invoke_progress_callbacks(callbacks, bytes_transferred):
    """Calls all progress callbacks

    :param callbacks: A list of progress callbacks to invoke
    :param bytes_transferred: The number of bytes transferred. This is passed
        to the callbacks. If no bytes were transferred the callbacks will not
        be invoked because no progress was achieved. It is also possible
        to receive a negative amount which comes from retrying a transfer
        request.
    """
    if bytes_transferred:
        for callback in callbacks:
            callback(bytes_transferred=bytes_transferred)