import gzip
import os
import shutil
import zipfile
from oslo_log import log as logging
from oslo_utils import encodeutils
from taskflow.patterns import linear_flow as lf
from taskflow import task
def header_lengths():
    headers = []
    for key, val in MAGIC_NUMBERS.items():
        offset, key = key.split('_')
        headers.append(int(offset) + len(val))
    return headers