from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def test_complete_no_code(self):
    output = FakeStdout()
    sot = complete.CompleteNoCode('doesNotMatter', output)
    sot.write(*self.given_cmdo_data())
    self.then_data(output.content)