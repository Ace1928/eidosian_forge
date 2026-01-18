from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@versionutils.deprecated(as_of=versionutils.deprecated.ICEHOUSE)
def outdated_method(self, *args):
    return args