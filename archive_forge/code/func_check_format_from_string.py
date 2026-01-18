from __future__ import unicode_literals
import os
import posixpath
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp
import pytest
from pybtex import errors, io
from .utils import diff, get_data
def check_format_from_string(engine, filenames):
    filenames_by_suffix = group_by_suffix(filenames)
    engine_name = engine.__name__.rsplit('.', 1)[-1]
    if '.aux' in filenames_by_suffix:
        from io import StringIO
        from pybtex import auxfile
        aux_contents = StringIO(get_data(filenames_by_suffix['.aux']))
        auxdata = auxfile.parse_file(aux_contents)
        citations = auxdata.citations
        style = auxdata.style
    else:
        citations = '*'
        style = posixpath.splitext(filenames_by_suffix['.bst'])[0]
    with cd_tempdir():
        copy_file(filenames_by_suffix['.bst'])
        bib_name = posixpath.splitext(filenames_by_suffix['.bib'])[0]
        bib_string = get_data(filenames_by_suffix['.bib'])
        with errors.capture():
            result = engine.format_from_string(bib_string, style=style, citations=citations)
        correct_result_name = '{0}_{1}.{2}.bbl'.format(bib_name, style, engine_name)
        correct_result = get_data(correct_result_name).replace('\r\n', '\n')
        assert result == correct_result, diff(correct_result, result)