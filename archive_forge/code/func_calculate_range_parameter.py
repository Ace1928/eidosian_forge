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
def calculate_range_parameter(part_size, part_index, num_parts, total_size=None):
    """Calculate the range parameter for multipart downloads/copies

    :type part_size: int
    :param part_size: The size of the part

    :type part_index: int
    :param part_index: The index for which this parts starts. This index starts
        at zero

    :type num_parts: int
    :param num_parts: The total number of parts in the transfer

    :returns: The value to use for Range parameter on downloads or
        the CopySourceRange parameter for copies
    """
    start_range = part_index * part_size
    if part_index == num_parts - 1:
        end_range = ''
        if total_size is not None:
            end_range = str(total_size - 1)
    else:
        end_range = start_range + part_size - 1
    range_param = 'bytes=%s-%s' % (start_range, end_range)
    return range_param