from __future__ import absolute_import, division, print_function
import unittest
from datashader import datashape
from datashader.datashape import lexer
def check_isolated_token(self, ds_str, tname, val=None):
    tid = getattr(lexer, tname)
    self.assertEqual(list(lexer.lex(ds_str)), [lexer.Token(tid, tname, (0, len(ds_str)), val)])