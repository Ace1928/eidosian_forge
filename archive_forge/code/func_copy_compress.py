import datetime
import decimal
import glob
import numbers
import os
import shutil
import string
from functools import partial
from stat import ST_DEV, ST_INO
from . import _string_parsers as string_parsers
from ._ctime_functions import get_ctime, set_ctime
from ._datetime import aware_now
@staticmethod
def copy_compress(path_in, path_out, opener, **kwargs):
    with open(path_in, 'rb') as f_in:
        with opener(path_out, **kwargs) as f_out:
            shutil.copyfileobj(f_in, f_out)