import pickle
import pytest
from urllib3.connectionpool import HTTPConnectionPool
from urllib3.exceptions import (
def test_header_parsing_errors(self):
    hpe = HeaderParsingError('defects', 'unparsed_data')
    assert 'defects' in str(hpe)
    assert 'unparsed_data' in str(hpe)