from __future__ import unicode_literals
import os
import posixpath
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp
import pytest
from pybtex import errors, io
from .utils import diff, get_data
def check_make_bibliography(engine, filenames):
    allowed_exts = {'.bst', '.bib', '.aux'}
    filenames_by_ext = dict(((posixpath.splitext(filename)[1], filename) for filename in filenames))
    engine_name = engine.__name__.rsplit('.', 1)[-1]
    for ext in filenames_by_ext:
        if ext not in allowed_exts:
            raise ValueError(ext)
    with cd_tempdir():
        copy_files(filenames)
        bib_name = posixpath.splitext(filenames_by_ext['.bib'])[0]
        bst_name = posixpath.splitext(filenames_by_ext['.bst'])[0]
        if '.aux' not in filenames_by_ext:
            write_aux('test.aux', bib_name, bst_name)
            filenames_by_ext['.aux'] = 'test.aux'
        with errors.capture():
            engine.make_bibliography(filenames_by_ext['.aux'])
        result_name = posixpath.splitext(filenames_by_ext['.aux'])[0] + '.bbl'
        with io.open_unicode(result_name) as result_file:
            result = result_file.read()
        correct_result_name = '{0}_{1}.{2}.bbl'.format(bib_name, bst_name, engine_name)
        correct_result = get_data(correct_result_name).replace('\r\n', '\n')
        assert result == correct_result, diff(correct_result, result)