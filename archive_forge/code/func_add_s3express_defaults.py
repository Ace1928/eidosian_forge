import functools
import logging
import math
import os
import random
import socket
import stat
import string
import threading
from collections import defaultdict
from botocore.exceptions import (
from botocore.httpchecksum import AwsChunkedWrapper
from botocore.utils import is_s3express_bucket
from s3transfer.compat import SOCKET_ERROR, fallocate, rename_file
def add_s3express_defaults(bucket, extra_args):
    if is_s3express_bucket(bucket) and 'ChecksumAlgorithm' not in extra_args:
        extra_args['ChecksumAlgorithm'] = 'crc32'