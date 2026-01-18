import os
import zstandard
import ujson as json
import time
import tarfile
import codecs
from functools import reduce
import jsonlines
import io
from zipfile import ZipFile
import gzip
from math import ceil
import mmap
import multiprocessing as mp
from pathlib import Path
def listdir_or_file(x):
    return list(filter(has_valid_extension, _listdir_or_file(x)))