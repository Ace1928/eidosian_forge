import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_encoding_substitution(self):
    meta_tag = '<meta content="text/html; charset=x-sjis" http-equiv="Content-type"/>'
    soup = self.soup(meta_tag)
    assert soup.meta['content'] == 'text/html; charset=x-sjis'
    utf_8 = soup.encode('utf-8')
    assert b'charset=utf-8' in utf_8
    euc_jp = soup.encode('euc_jp')
    assert b'charset=euc_jp' in euc_jp
    shift_jis = soup.encode('shift-jis')
    assert b'charset=shift-jis' in shift_jis
    utf_16_u = soup.encode('utf-16').decode('utf-16')
    assert 'charset=utf-16' in utf_16_u