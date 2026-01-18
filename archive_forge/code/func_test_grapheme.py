from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_grapheme(self):
    self.assertEqual(regex.match('\\X', 'à').span(), (0, 1))
    self.assertEqual(regex.match('\\X', 'à').span(), (0, 2))
    self.assertEqual(regex.findall('\\X', 'aààeéé'), ['a', 'à', 'à', 'e', 'é', 'é'])
    self.assertEqual(regex.findall('\\X{3}', 'aààeéé'), ['aàà', 'eéé'])
    self.assertEqual(regex.findall('\\X', '\r\r\ńÁ'), ['\r', '\r\n', '́', 'Á'])