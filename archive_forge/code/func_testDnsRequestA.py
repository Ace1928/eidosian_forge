import DNS
import unittest
def testDnsRequestA(self):
    dnsobj = DNS.DnsRequest('example.org')
    a_response = dnsobj.qry(qtype='A', resulttype='text', timeout=1)
    self.assertTrue(a_response.answers)
    self.assertEqual(a_response.answers[0]['data'].count('.'), 3)
    self.assertEqual(a_response.answers[0]['data'], '93.184.216.34')
    ad_response = dnsobj.qry(qtype='A', timeout=1)
    self.assertTrue(ad_response.answers)
    self.assertEqual(ad_response.answers[0]['data'], ipaddress.IPv4Address('93.184.216.34'))
    ab_response = dnsobj.qry(qtype='A', resulttype='binary', timeout=1)
    self.assertTrue(ab_response.answers)
    self.assertEqual(len(ab_response.answers[0]['data']), 4)
    for b in ab_response.answers[0]['data']:
        assertIsByte(b)
    self.assertEqual(ab_response.answers[0]['data'], b']\xb8\xd8"')
    ai_response = dnsobj.qry(qtype='A', resulttype='integer', timeout=1)
    self.assertTrue(ai_response.answers)
    self.assertEqual(ai_response.answers[0]['data'], 1572395042)