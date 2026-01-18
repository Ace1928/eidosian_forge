import unittest
from pyparsing import ParseException
from .btpyparse import Macro
from . import btpyparse as bp
def test_bibfile(self):
    txt = '@some_entry{akey, aname = "about something",\n        another={something else}}'
    res = bp.bibfile.parseString(txt)
    self.assertEqual(res.asList(), [['some_entry', 'akey', ['aname', 'about something'], ['another', 'something else']]])