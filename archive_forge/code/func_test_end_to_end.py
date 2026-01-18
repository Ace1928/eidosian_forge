import sys
from parser import parse, MalformedQueryStringError
from builder import build
import unittest
def test_end_to_end(self):
    parsed = parse('a[]=1&a[]=2')
    result = build(parsed)
    self.assertEquals(result, 'a[]=1&a[]=2')