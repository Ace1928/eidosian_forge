import os
import sys
import time
import shutil
import platform
import tempfile
import unittest
import multiprocessing
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
def remove_tmp_file(self, tmppath):
    try:
        os.unlink(tmppath)
    except Exception as e:
        msg = str(e)
        if 'being used by another process' in msg and platform.system().lower() == 'windows':
            return
        raise e