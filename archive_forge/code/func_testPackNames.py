import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def testPackNames(self):
    from DNS.Lib import Packer
    for namelist, result in self.knownPackValues:
        p = Packer()
        for n in namelist:
            p.addname(n)
        self.assertEqual(p.getbuf(), result)