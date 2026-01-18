import pytest
from bs4.element import (
from . import SoupTest
def test_cdata_is_never_formatted(self):
    """Text inside a CData object is passed into the formatter.

        But the return value is ignored.
        """
    self.count = 0

    def increment(*args):
        self.count += 1
        return 'BITTER FAILURE'
    soup = self.soup('')
    cdata = CData('<><><>')
    soup.insert(1, cdata)
    assert b'<![CDATA[<><><>]]>' == soup.encode(formatter=increment)
    assert 1 == self.count