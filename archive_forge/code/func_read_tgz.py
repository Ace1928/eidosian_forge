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
def read_tgz(self, file):
    gz = gzip.open(file)
    yield from (x.decode('utf-8') for x in tarfile_reader(gz, streaming=False))