import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_handle_json_from_file_valid_file(self):
    contents = '{"step": "upgrade", "interface": "deploy"}'
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(contents)
        f.flush()
        steps = utils.handle_json_from_file(f.name)
    self.assertEqual(jsonutils.loads(contents), steps)