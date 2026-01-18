import testtools
from oslotest import base
from octavia_lib.hacking import checks
def test_check_no_logging_imports(self):
    f = checks.check_no_logging_imports
    self.assertLinePasses(f, 'from oslo_log import log')
    self.assertLineFails(f, 'from logging import log')
    self.assertLineFails(f, 'import logging')