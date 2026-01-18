from __future__ import with_statement
import textwrap
from difflib import ndiff
from io import open
from os import listdir
from os.path import dirname, isdir, join, realpath, relpath, splitext
import pytest
import chardet
def gen_test_params():
    """Yields tuples of paths and encodings to use for test_encoding_detection"""
    base_path = relpath(join(dirname(realpath(__file__)), 'tests'))
    for encoding in listdir(base_path):
        path = join(base_path, encoding)
        if not isdir(path):
            continue
        encoding = encoding.lower()
        for postfix in ['-arabic', '-bulgarian', '-cyrillic', '-greek', '-hebrew', '-hungarian', '-turkish']:
            if encoding.endswith(postfix):
                encoding = encoding.rpartition(postfix)[0]
                break
        if encoding in MISSING_ENCODINGS:
            continue
        for file_name in listdir(path):
            ext = splitext(file_name)[1].lower()
            if ext not in ['.html', '.txt', '.xml', '.srt']:
                continue
            full_path = join(path, file_name)
            test_case = (full_path, encoding)
            if full_path in EXPECTED_FAILURES:
                test_case = pytest.mark.xfail(test_case)
            yield test_case