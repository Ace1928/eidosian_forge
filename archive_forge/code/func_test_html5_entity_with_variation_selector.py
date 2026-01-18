import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_html5_entity_with_variation_selector(self):
    data = 'fjords ⊔ penguins'
    markup = 'fjords &sqcup; penguins'
    assert self.sub.substitute_html(data) == markup
    data = 'fjords ⊔︀ penguins'
    markup = 'fjords &sqcups; penguins'
    assert self.sub.substitute_html(data) == markup