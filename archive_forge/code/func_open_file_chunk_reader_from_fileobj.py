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
def open_file_chunk_reader_from_fileobj(self, fileobj, chunk_size, full_file_size, callbacks, close_callbacks=None):
    return ReadFileChunk(fileobj, chunk_size, full_file_size, callbacks=callbacks, enable_callbacks=False, close_callbacks=close_callbacks)