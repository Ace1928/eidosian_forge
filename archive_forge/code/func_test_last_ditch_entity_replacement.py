import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_last_ditch_entity_replacement(self):
    doc = b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?>\n<html><b>\xd8\xa8\xd8\xaa\xd8\xb1</b>\n<i>\xc8\xd2\xd1\x90\xca\xd1\xed\xe4</i></html>'
    chardet = bs4.dammit.chardet_dammit
    logging.disable(logging.WARNING)
    try:

        def noop(str):
            return None
        bs4.dammit.chardet_dammit = noop
        dammit = UnicodeDammit(doc)
        assert True == dammit.contains_replacement_characters
        assert '�' in dammit.unicode_markup
        soup = BeautifulSoup(doc, 'html.parser')
        assert soup.contains_replacement_characters
    finally:
        logging.disable(logging.NOTSET)
        bs4.dammit.chardet_dammit = chardet