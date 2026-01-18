import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def testUnpacker(self):
    if self.doUnpack is not None:
        if hasattr(self.__class__, 'doUnpack') and hasattr(self, 'packerExpectedResult'):
            u = self.RRunpacker(self.packerExpectedResult)
            rrbits = u.getRRheader()[:4]
            specbits = self.doUnpack(u)
            try:
                u.endRR()
            except DNS.Lib.UnpackError:
                self.assertEqual(0, 'Not at end of RR!')
            return self.checkUnpackResult(rrbits, specbits)
        else:
            me = self.__class__.__name__
            if me != 'PackerTestCase':
                self.assertEquals(self.__class__.__name__, 'Unpack NotImplemented')