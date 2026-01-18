import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def test16bitUnpacking(self):
    """ unpack16bit should give known output for known input """
    for i, s in self.knownValues:
        result = DNS.Lib.unpack16bit(s)
        self.assertEqual(i, result)