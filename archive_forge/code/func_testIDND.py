import DNS
import unittest
def testIDND(self):
    """Can we lookup an internationalized domain name?"""
    dnsob = DNS.DnsRequest('xn--bb-eka.at')
    unidnsob = DNS.DnsRequest('öbb.at')
    a_resp = dnsob.req(qtype='A', resulttype='text', timeout=1)
    ua_resp = unidnsob.req(qtype='A', resulttype='text', timeout=1)
    self.assertTrue(a_resp.answers)
    self.assertTrue(ua_resp.answers)
    self.assertEqual(ua_resp.answers[0]['data'], a_resp.answers[0]['data'])