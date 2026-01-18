import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def testIPaddrPacking(self):
    """ addr2bin should give known output for known input """
    for i, s in self.knownValues:
        result = DNS.Lib.addr2bin(i)
        self.assertEqual(s, result)