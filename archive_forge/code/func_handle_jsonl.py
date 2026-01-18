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
def handle_jsonl(jsonl_reader, get_meta, autojoin_paragraphs, para_joiner, key='text'):
    for ob in jsonl_reader:
        if isinstance(ob, str):
            assert not get_meta
            yield ob
            continue
        text = ob[key]
        if autojoin_paragraphs and isinstance(text, list):
            text = para_joiner.join(text)
        if get_meta:
            yield (text, ob['meta'] if 'meta' in ob else {})
        else:
            yield text