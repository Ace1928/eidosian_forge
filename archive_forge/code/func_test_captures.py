from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_captures(self):
    self.assertEqual(regex.search('(\\w)+', 'abc').captures(1), ['a', 'b', 'c'])
    self.assertEqual(regex.search('(\\w{3})+', 'abcdef').captures(0, 1), (['abcdef'], ['abc', 'def']))
    self.assertEqual(regex.search('^(\\d{1,3})(?:\\.(\\d{1,3})){3}$', '192.168.0.1').captures(1, 2), (['192'], ['168', '0', '1']))
    self.assertEqual(regex.match('^([0-9A-F]{2}){4} ([a-z]\\d){5}$', '3FB52A0C a2c4g3k9d3').captures(1, 2), (['3F', 'B5', '2A', '0C'], ['a2', 'c4', 'g3', 'k9', 'd3']))
    self.assertEqual(regex.match('([a-z]W)([a-z]X)+([a-z]Y)', 'aWbXcXdXeXfY').captures(1, 2, 3), (['aW'], ['bX', 'cX', 'dX', 'eX'], ['fY']))
    self.assertEqual(regex.search('.*?(?=(.)+)b', 'ab').captures(1), ['b'])
    self.assertEqual(regex.search('.*?(?>(.){0,2})d', 'abcd').captures(1), ['b', 'c'])
    self.assertEqual(regex.search('(.)+', 'a').captures(1), ['a'])