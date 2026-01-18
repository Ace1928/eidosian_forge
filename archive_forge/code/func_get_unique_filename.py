import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def get_unique_filename(test_name: str='test', kwargs: dict={}, extension: str='csv', data_dir: Union[str, Path]='', suffix: str='', debug_mode=False):
    """Returns unique file name with specified parameters.

    Parameters
    ----------
    test_name: str
        name of the test for which the unique file name is needed.
    kwargs: list of ints
        Unique combiantion of test parameters for creation of unique name.
    extension: str, default: "csv"
        Extension of unique file.
    data_dir: Union[str, Path]
        Data directory where test files will be created.
    suffix: str
        String to append to the resulted name.
    debug_mode: bool, default: False
        Get unique filename containing kwargs values.
        Otherwise kwargs values will be replaced with hash equivalent.

    Returns
    -------
        Unique file name.
    """
    suffix_part = f'_{suffix}' if suffix else ''
    extension_part = f'.{extension}' if extension else ''
    if debug_mode:
        if len(kwargs) == 0 and extension == 'csv' and (suffix == ''):
            return os.path.join(data_dir, test_name + suffix_part + f'.{extension}')
        assert '.' not in extension, "please provide pure extension name without '.'"
        prohibited_chars = ['"', '\n']
        non_prohibited_char = 'np_char'
        char_counter = 0
        kwargs_name = dict(kwargs)
        for key, value in kwargs_name.items():
            for char in prohibited_chars:
                if isinstance(value, str) and char in value or callable(value):
                    kwargs_name[key] = non_prohibited_char + str(char_counter)
                    char_counter += 1
        parameters_values = '_'.join([str(value) if not isinstance(value, (list, tuple)) else '_'.join([str(x) for x in value]) for value in kwargs_name.values()])
        return os.path.join(data_dir, test_name + parameters_values + suffix_part + extension_part)
    else:
        import uuid
        return os.path.join(data_dir, uuid.uuid1().hex + suffix_part + extension_part)