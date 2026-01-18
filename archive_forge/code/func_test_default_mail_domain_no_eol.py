import os
import sys
from .. import bedding, osutils, tests
def test_default_mail_domain_no_eol(self):
    with open('no_eol', 'w') as f:
        f.write('domainname.com')
    r = bedding._get_default_mail_domain('no_eol')
    self.assertEqual('domainname.com', r)