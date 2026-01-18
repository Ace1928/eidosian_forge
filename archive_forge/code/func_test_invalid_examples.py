import unittest
from traits.observation._generated_parser import (
def test_invalid_examples(self):
    bad_examples = ['', '1name', 'a.b.c^abc', '[a.b]c', 'a*.c', 'a:[b,c]:', '.a', 'a()', '-a']
    for bad_example in bad_examples:
        with self.subTest(bad_example=bad_example):
            with self.assertRaises(UnexpectedInput):
                PARSER.parse(bad_example)