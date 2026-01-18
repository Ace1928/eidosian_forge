import DNS
import unittest
def testDnsRequestSrvD(self):
    dnsob = DNS.Request(qtype='srv')
    respdef = dnsob.req('_ldap._tcp.openldap.org', timeout=1)
    self.assertTrue(respdef.answers)
    data = respdef.answers[0]['data']
    self.assertEqual(len(data), 4)
    self.assertEqual(data[2], 389)
    self.assertTrue('openldap.org' in data[3])