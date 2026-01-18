import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def fields_to_entry_n_parents(fields, _int=int):
    path_name_file_id_key = (fields[0], fields[1], fields[2])
    trees = [(fields[cur], fields[cur + 1], _int(fields[cur + 2]), fields[cur + 3] == b'y', fields[cur + 4]) for cur in range(3, len(fields) - 1, 5)]
    return (path_name_file_id_key, trees)