from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def test_complete_bash(self):
    output = FakeStdout()
    sot = complete.CompleteBash('openstack', output)
    sot.write(*self.given_cmdo_data())
    self.then_data(output.content)
    self.assertIn('_openstack()\n', output.content[0])
    self.assertIn('complete -F _openstack openstack\n', output.content[-1])