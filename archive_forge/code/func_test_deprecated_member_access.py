import warnings
from bs4.element import (
from . import SoupTest
def test_deprecated_member_access(self):
    soup = self.soup('<b><i></i></b>')
    with warnings.catch_warnings(record=True) as w:
        tag = soup.bTag
    assert soup.b == tag
    assert '.bTag is deprecated, use .find("b") instead. If you really were looking for a tag called bTag, use .find("bTag")' == str(w[0].message)