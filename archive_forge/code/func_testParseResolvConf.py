import DNS
import unittest
def testParseResolvConf(self):
    DNS.defaults['server'] = []
    if 'domain' in DNS.defaults:
        del DNS.defaults['domain']
    self.assertEqual(len(DNS.defaults['server']), 0)
    resolv = ['# a comment', 'domain example.org', 'nameserver 127.0.0.1']
    DNS.ParseResolvConfFromIterable(resolv)
    self.assertEqual(len(DNS.defaults['server']), 1)
    self.assertEqual(DNS.defaults['server'][0], '127.0.0.1')
    self.assertEqual(DNS.defaults['domain'], 'example.org')