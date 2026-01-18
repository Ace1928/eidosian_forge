import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_html5_entity(self):
    for entity, u in (('&models;', '⊧'), ('&Nfr;', '𝔑'), ('&ngeqq;', '≧̸'), ('&not;', '¬'), ('&Not;', '⫬'), '||', ('fj', 'fj'), ('&gt;', '>'), ('&lt;', '<'), ('&amp;', '&')):
        template = '3 %s 4'
        raw = template % u
        with_entities = template % entity
        assert self.sub.substitute_html(raw) == with_entities