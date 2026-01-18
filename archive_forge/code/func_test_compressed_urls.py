from io import BytesIO
import logging
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.feather_format import read_feather
from pandas.io.parsers import read_csv
@pytest.mark.network
@pytest.mark.single_cpu
@pytest.mark.parametrize('mode', ['explicit', 'infer'])
@pytest.mark.parametrize('engine', ['python', 'c'])
def test_compressed_urls(httpserver, datapath, salaries_table, mode, engine, compression_only, compression_to_extension):
    if compression_only == 'tar':
        pytest.skip('TODO: Add tar salaraies.csv to pandas/io/parsers/data')
    extension = compression_to_extension[compression_only]
    with open(datapath('io', 'parser', 'data', 'salaries.csv' + extension), 'rb') as f:
        httpserver.serve_content(content=f.read())
    url = httpserver.url + '/salaries.csv' + extension
    if mode != 'explicit':
        compression_only = mode
    url_table = read_csv(url, sep='\t', compression=compression_only, engine=engine)
    tm.assert_frame_equal(url_table, salaries_table)