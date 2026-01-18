import codecs
import sys
import unittest
import idna.codec
def testIncrementalEncoder(self):
    incremental_tests = (('python.org', b'python.org'), ('python.org.', b'python.org.'), ('pythön.org', b'xn--pythn-mua.org'), ('pythön.org.', b'xn--pythn-mua.org.'))
    for decoded, encoded in incremental_tests:
        self.assertEqual(b''.join(codecs.iterencode(decoded, 'idna')), encoded)
    encoder = codecs.getincrementalencoder('idna')()
    self.assertEqual(encoder.encode('äx'), b'')
    self.assertEqual(encoder.encode('ample.org'), b'xn--xample-9ta.')
    self.assertEqual(encoder.encode('', True), b'org')
    encoder.reset()
    self.assertEqual(encoder.encode('äx'), b'')
    self.assertEqual(encoder.encode('ample.org.'), b'xn--xample-9ta.org.')
    self.assertEqual(encoder.encode('', True), b'')