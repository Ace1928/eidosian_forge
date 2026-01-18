from io import BytesIO
import pytest
import logging
from matplotlib import _afm
from matplotlib import font_manager as fm
def test_get_familyname_guessed():
    fh = BytesIO(AFM_TEST_DATA)
    font = _afm.AFM(fh)
    del font._header[b'FamilyName']
    assert font.get_familyname() == 'My Font'