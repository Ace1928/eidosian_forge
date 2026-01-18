import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_known_definite_versus_user_encodings(self):
    data = b'\xff\xfe<\x00a\x00>\x00\xe1\x00\xe9\x00<\x00/\x00a\x00>\x00'
    dammit = UnicodeDammit(data)
    before = UnicodeDammit(data, known_definite_encodings=['utf-16'])
    assert 'utf-16' == before.original_encoding
    after = UnicodeDammit(data, user_encodings=['utf-8'])
    assert 'utf-16le' == after.original_encoding
    assert ['utf-16le'] == [x[0] for x in dammit.tried_encodings]
    hebrew = b'\xed\xe5\xec\xf9'
    dammit = UnicodeDammit(hebrew, known_definite_encodings=['utf-8'], user_encodings=['iso-8859-8'])
    assert 'iso-8859-8' == dammit.original_encoding
    assert ['utf-8', 'iso-8859-8'] == [x[0] for x in dammit.tried_encodings]