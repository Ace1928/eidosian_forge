import os
import sys
from .. import bedding, osutils, tests
def test_default_mail_domain_multiple_lines(self):
    with open('multiple_lines', 'w') as f:
        f.write('domainname.com\nsome other text\n')
    r = bedding._get_default_mail_domain('multiple_lines')
    self.assertEqual('domainname.com', r)