import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
def test_command_echo_hello(self):
    command = ['sh', '-c', 'echo hello']
    result = utils.parse_command(command)
    self.assertEqual('"sh" "-c" "echo hello"', result)