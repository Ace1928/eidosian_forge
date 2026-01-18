import pathlib
import pytest
from ..filename_parser import (
def test__stringify_path():
    res = _stringify_path('fname.ext.gz')
    assert res == 'fname.ext.gz'
    res = _stringify_path(pathlib.Path('fname.ext.gz'))
    assert res == 'fname.ext.gz'
    home = pathlib.Path.home().as_posix()
    res = _stringify_path(pathlib.Path('~/fname.ext.gz'))
    assert res == f'{home}/fname.ext.gz'
    res = _stringify_path(pathlib.Path('./fname.ext.gz'))
    assert res == 'fname.ext.gz'
    res = _stringify_path(pathlib.Path('../fname.ext.gz'))
    assert res == '../fname.ext.gz'