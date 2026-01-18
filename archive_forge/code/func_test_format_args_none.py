import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_format_args_none(self):
    self.assertEqual({}, utils.format_args(None))