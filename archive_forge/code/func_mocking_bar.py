import mock # Yes, we only test the rolling backport
import testtools
from fixtures import (
def mocking_bar(self):
    return 'mocked!'