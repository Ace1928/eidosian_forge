import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def testWithNoGetData(self):
    u = DNS.Lib.RRunpacker(self.packerCorrect)
    u.getRRheader()
    self.assertRaises(DNS.Lib.UnpackError, u.endRR)