from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@versionutils.deprecated(as_of=versionutils.deprecated.GRIZZLY, in_favor_of='different_stuff()', remove_in=0)
def do_outdated_stuff():
    return