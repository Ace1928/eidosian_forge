from __future__ import annotations
from collections.abc import (
from typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
def get_default_engine(ext: str, mode: Literal['reader', 'writer']='reader') -> str:
    """
    Return the default reader/writer for the given extension.

    Parameters
    ----------
    ext : str
        The excel file extension for which to get the default engine.
    mode : str {'reader', 'writer'}
        Whether to get the default engine for reading or writing.
        Either 'reader' or 'writer'

    Returns
    -------
    str
        The default engine for the extension.
    """
    _default_readers = {'xlsx': 'openpyxl', 'xlsm': 'openpyxl', 'xlsb': 'pyxlsb', 'xls': 'xlrd', 'ods': 'odf'}
    _default_writers = {'xlsx': 'openpyxl', 'xlsm': 'openpyxl', 'xlsb': 'pyxlsb', 'ods': 'odf'}
    assert mode in ['reader', 'writer']
    if mode == 'writer':
        xlsxwriter = import_optional_dependency('xlsxwriter', errors='warn')
        if xlsxwriter:
            _default_writers['xlsx'] = 'xlsxwriter'
        return _default_writers[ext]
    else:
        return _default_readers[ext]