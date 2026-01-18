from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bytes_str_mixing(self):
    pat = regex.compile('.')
    bpat = regex.compile(b'.')
    self.assertRaisesRegex(TypeError, self.STR_PAT_ON_BYTES, lambda: pat.match(b'b'))
    self.assertRaisesRegex(TypeError, self.BYTES_PAT_ON_STR, lambda: bpat.match('b'))
    self.assertRaisesRegex(TypeError, self.STR_PAT_BYTES_TEMPL, lambda: pat.sub(b'b', 'c'))
    self.assertRaisesRegex(TypeError, self.STR_PAT_ON_BYTES, lambda: pat.sub('b', b'c'))
    self.assertRaisesRegex(TypeError, self.STR_PAT_ON_BYTES, lambda: pat.sub(b'b', b'c'))
    self.assertRaisesRegex(TypeError, self.BYTES_PAT_ON_STR, lambda: bpat.sub(b'b', 'c'))
    self.assertRaisesRegex(TypeError, self.BYTES_PAT_STR_TEMPL, lambda: bpat.sub('b', b'c'))
    self.assertRaisesRegex(TypeError, self.BYTES_PAT_ON_STR, lambda: bpat.sub('b', 'c'))
    self.assertRaisesRegex(ValueError, self.BYTES_PAT_UNI_FLAG, lambda: regex.compile(b'\\w', regex.UNICODE))
    self.assertRaisesRegex(ValueError, self.BYTES_PAT_UNI_FLAG, lambda: regex.compile(b'(?u)\\w'))
    self.assertRaisesRegex(ValueError, self.MIXED_FLAGS, lambda: regex.compile('\\w', regex.UNICODE | regex.ASCII))
    self.assertRaisesRegex(ValueError, self.MIXED_FLAGS, lambda: regex.compile('(?u)\\w', regex.ASCII))
    self.assertRaisesRegex(ValueError, self.MIXED_FLAGS, lambda: regex.compile('(?a)\\w', regex.UNICODE))
    self.assertRaisesRegex(ValueError, self.MIXED_FLAGS, lambda: regex.compile('(?au)\\w'))