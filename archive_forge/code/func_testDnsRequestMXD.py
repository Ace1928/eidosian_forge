import DNS
import unittest
def testDnsRequestMXD(self):
    dnsob = DNS.DnsRequest('ietf.org')
    mx_response = dnsob.req(qtype='MX', timeout=1)
    self.assertTrue(mx_response.answers[0])
    self.assertEqual(mx_response.answers[0]['data'], (0, 'mail.ietf.org'))
    m = DNS.mxlookup('ietf.org', timeout=1)
    self.assertEqual(mx_response.answers[0]['data'], m[0])