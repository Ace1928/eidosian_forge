import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def dest_is_valid_page(obj, page_object_nums, pagecount):
    num = mupdf.pdf_to_num(obj)
    if num == 0:
        return 0
    for i in range(pagecount):
        if page_object_nums[i] == num:
            return 1
    return 0