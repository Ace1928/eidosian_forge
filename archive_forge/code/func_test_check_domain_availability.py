import boto
from boto.route53.domains.exceptions import InvalidInput
from tests.compat import unittest
def test_check_domain_availability(self):
    response = self.route53domains.check_domain_availability(domain_name='amazon.com', idn_lang_code='eng')
    self.assertEqual(response, {'Availability': 'UNAVAILABLE'})