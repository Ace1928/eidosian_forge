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
def tarfile_reader(file, streaming=False):
    offset = 0
    paxfilesize = None
    while True:
        hdr = file.read(512)
        offset += 512
        if hdr[124:135] == b'\x00' * 11:
            break
        fname = hdr[:100].split(b'\x00')[0]
        if paxfilesize is not None:
            size = paxfilesize
            paxfilesize = None
        else:
            size = int(hdr[124:135], 8)
        padded_size = ceil(size / 512) * 512
        type = chr(hdr[156])
        if type == 'x':
            meta = file.read(padded_size)[:size]

            def kv(x):
                return x.decode('utf-8').split(' ')[1].split('=')
            paxfileattrs = {kv(x)[0]: kv(x)[1] for x in meta.split(b'\n') if x}
            paxfilesize = int(paxfileattrs['size'])
            offset += padded_size
            continue
        elif type != '0' and type != '\x00':
            if streaming:
                file.seek(padded_size, os.SEEK_CUR)
            else:
                file.read(padded_size)
            offset += padded_size
            continue
        if streaming:
            if size != 0:
                mmo = mmap.mmap(file.fileno(), length=offset + size, access=mmap.ACCESS_READ)
                mmo.seek(offset)
                yield mmo
            file.seek(padded_size, os.SEEK_CUR)
        else:
            yield file.read(padded_size)[:size]
        offset += padded_size