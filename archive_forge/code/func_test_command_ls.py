import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_command_ls(self):
    command = ['ls', '-al']
    result = utils.parse_command(command)
    self.assertEqual('"ls" "-al"', result)