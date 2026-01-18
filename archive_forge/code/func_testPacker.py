import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def testPacker(self):
    p = self.RRpacker()
    check = self.doPack(p)
    if p is not None and check is not TestCompleted:
        return self.checkPackResult(p)