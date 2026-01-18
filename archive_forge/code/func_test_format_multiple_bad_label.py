import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_format_multiple_bad_label(self):
    labels = ['K1=V1', 'K22.2.2.2']
    ex = self.assertRaises(exc.CommandError, utils.format_labels, labels)
    self.assertEqual('labels must be a list of KEY=VALUE not K22.2.2.2', str(ex))