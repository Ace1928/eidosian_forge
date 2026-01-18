import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_format_labels_parse_comma_false(self):
    la = utils.format_labels(['K1=V1,K2=2.2.2.2,K=V'], parse_comma=False)
    self.assertEqual({'K1': 'V1,K2=2.2.2.2,K=V'}, la)