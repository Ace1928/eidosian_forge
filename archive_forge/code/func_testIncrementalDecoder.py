import codecs
import sys
import unittest
import idna.codec
def testIncrementalDecoder(self):
    incremental_tests = (('python.org', b'python.org'), ('python.org.', b'python.org.'), ('pythön.org', b'xn--pythn-mua.org'), ('pythön.org.', b'xn--pythn-mua.org.'))
    for decoded, encoded in incremental_tests:
        self.assertEqual(''.join(codecs.iterdecode((bytes([c]) for c in encoded), 'idna')), decoded)
    decoder = codecs.getincrementaldecoder('idna')()
    self.assertEqual(decoder.decode(b'xn--xam'), '')
    self.assertEqual(decoder.decode(b'ple-9ta.o'), 'äxample.')
    self.assertEqual(decoder.decode(b'rg'), '')
    self.assertEqual(decoder.decode(b'', True), 'org')
    decoder.reset()
    self.assertEqual(decoder.decode(b'xn--xam'), '')
    self.assertEqual(decoder.decode(b'ple-9ta.o'), 'äxample.')
    self.assertEqual(decoder.decode(b'rg.'), 'org.')
    self.assertEqual(decoder.decode(b'', True), '')