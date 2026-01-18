from __future__ import print_function
import sys
import pytest  # NOQA
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
@pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
def test_blank_line_after_literal_chip(self):
    s = '\n        c:\n        - |\n          This item\n          has a blank line\n          following it.\n\n        - |\n          To visually separate it from this item.\n\n          This item contains a blank line.\n\n\n        '
    d = round_trip_load(dedent(s))
    print(d)
    round_trip(s)
    assert d['c'][0].split('it.')[1] == '\n'
    assert d['c'][1].split('line.')[1] == '\n'