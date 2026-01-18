import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def pause():
    time.sleep(5.0)