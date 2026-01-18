import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_exclude_encodings(self):
    utf8_data = 'Räksmörgås'.encode('utf-8')
    dammit = UnicodeDammit(utf8_data, exclude_encodings=['utf-8'])
    assert dammit.original_encoding.lower() == 'windows-1252'
    dammit = UnicodeDammit(utf8_data, exclude_encodings=['utf-8', 'windows-1252'])
    assert dammit.original_encoding == None