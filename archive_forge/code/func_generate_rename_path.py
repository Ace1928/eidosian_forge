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
def generate_rename_path(root, ext, creation_time):
    creation_datetime = datetime.datetime.fromtimestamp(creation_time)
    date = FileDateFormatter(creation_datetime)
    renamed_path = '{}.{}{}'.format(root, date, ext)
    counter = 1
    while os.path.exists(renamed_path):
        counter += 1
        renamed_path = '{}.{}.{}{}'.format(root, date, counter, ext)
    return renamed_path