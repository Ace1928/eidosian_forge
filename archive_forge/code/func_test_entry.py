import unittest
from pyparsing import ParseException
from .btpyparse import Macro
from . import btpyparse as bp
def test_entry(self):
    txt = '@some_entry{akey, aname = "about something",\n        another={something else}}'
    res = bp.entry.parseString(txt)
    self.assertEqual(res.asList(), ['some_entry', 'akey', ['aname', 'about something'], ['another', 'something else']])
    txt = '@SOME_ENTRY{akey, ANAME = "about something",\n        another={something else}}'
    res = bp.entry.parseString(txt)
    self.assertEqual(res.asList(), ['some_entry', 'akey', ['aname', 'about something'], ['another', 'something else']])