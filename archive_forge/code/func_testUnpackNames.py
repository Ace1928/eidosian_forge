import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def testUnpackNames(self):
    from DNS.Lib import Unpacker
    for namelist, result in self.knownUnpackValues:
        u = Unpacker(result)
        names = []
        for i in range(len(namelist)):
            n = u.getname()
            names.append(n)
        self.assertEqual(names, namelist)