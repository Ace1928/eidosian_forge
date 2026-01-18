from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_basic_regex_sub(self):
    self.assertEqual(regex.sub('(?i)b+', 'x', 'bbbb BBBB'), 'x x')
    self.assertEqual(regex.sub('\\d+', self.bump_num, '08.2 -2 23x99y'), '9.3 -3 24x100y')
    self.assertEqual(regex.sub('\\d+', self.bump_num, '08.2 -2 23x99y', 3), '9.3 -3 23x99y')
    self.assertEqual(regex.sub('.', lambda m: '\\n', 'x'), '\\n')
    self.assertEqual(regex.sub('.', '\\n', 'x'), '\n')
    self.assertEqual(regex.sub('(?P<a>x)', '\\g<a>\\g<a>', 'xx'), 'xxxx')
    self.assertEqual(regex.sub('(?P<a>x)', '\\g<a>\\g<1>', 'xx'), 'xxxx')
    self.assertEqual(regex.sub('(?P<unk>x)', '\\g<unk>\\g<unk>', 'xx'), 'xxxx')
    self.assertEqual(regex.sub('(?P<unk>x)', '\\g<1>\\g<1>', 'xx'), 'xxxx')
    self.assertEqual(regex.sub('a', '\\t\\n\\v\\r\\f\\a\\b', 'a'), '\t\n\x0b\r\x0c\x07\x08')
    self.assertEqual(regex.sub('a', '\t\n\x0b\r\x0c\x07', 'a'), '\t\n\x0b\r\x0c\x07')
    self.assertEqual(regex.sub('a', '\t\n\x0b\r\x0c\x07', 'a'), chr(9) + chr(10) + chr(11) + chr(13) + chr(12) + chr(7))
    self.assertEqual(regex.sub('^\\s*', 'X', 'test'), 'Xtest')
    self.assertEqual(regex.sub('x', '\\x0A', 'x'), '\n')
    self.assertEqual(regex.sub('x', '\\u000A', 'x'), '\n')
    self.assertEqual(regex.sub('x', '\\U0000000A', 'x'), '\n')
    self.assertEqual(regex.sub('x', '\\N{LATIN CAPITAL LETTER A}', 'x'), 'A')
    self.assertEqual(regex.sub(b'x', b'\\x0A', b'x'), b'\n')