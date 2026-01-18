import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_copy_preserves_builder_information(self):
    tag = self.soup('<p></p>').p
    tag.sourceline = 10
    tag.sourcepos = 33
    copied = tag.__copy__()
    assert tag.sourceline == copied.sourceline
    assert tag.sourcepos == copied.sourcepos
    assert tag.can_be_empty_element == copied.can_be_empty_element
    assert tag.cdata_list_attributes == copied.cdata_list_attributes
    assert tag.preserve_whitespace_tags == copied.preserve_whitespace_tags
    assert tag.interesting_string_types == copied.interesting_string_types