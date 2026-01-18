import DNS
import unittest
def testDkimRequest(self):
    q = '20161025._domainkey.google.com'
    dnsobj = DNS.Request(q, qtype='txt')
    resp = dnsobj.qry(timeout=1)
    self.assertTrue(resp.answers)
    data = resp.answers[0]['data']
    self.assertFalse(isinstance(data[0], str))
    self.assertTrue(data[0].startswith(b'k=rsa'))