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
def insert_lines_to_csv(csv_name: str, lines_positions: list, lines_type: str='blank', encoding: str=None, **csv_reader_writer_params):
    """Insert lines to ".csv" file.

    Parameters
    ----------
    csv_name: str
        ".csv" file that should be modified.
    lines_positions: list of ints
        Lines postions that sghould be modified (serial number
        of line - begins from 0, ends in <rows_number> - 1).
    lines_type: str
        Lines types that should be inserted to ".csv" file. Possible types:
        "blank" - empty line without any delimiters/separators,
        "bad" - lines with len(lines_data) > cols_number
    encoding: str
        Encoding type that should be used during file reading and writing.
    """
    if lines_type == 'blank':
        lines_data = []
    elif lines_type == 'bad':
        cols_number = len(pandas.read_csv(csv_name, nrows=1).columns)
        lines_data = [x for x in range(cols_number + 1)]
    else:
        raise ValueError(f"acceptable values for  parameter are ['blank', 'bad'], actually passed {lines_type}")
    lines = []
    with open(csv_name, 'r', encoding=encoding, newline='') as read_file:
        try:
            dialect = csv.Sniffer().sniff(read_file.read())
            read_file.seek(0)
        except Exception:
            dialect = None
        reader = csv.reader(read_file, dialect=dialect if dialect is not None else 'excel', **csv_reader_writer_params)
        counter = 0
        for row in reader:
            if counter in lines_positions:
                lines.append(lines_data)
            else:
                lines.append(row)
            counter += 1
    with open(csv_name, 'w', encoding=encoding, newline='') as write_file:
        writer = csv.writer(write_file, dialect=dialect if dialect is not None else 'excel', **csv_reader_writer_params)
        writer.writerows(lines)