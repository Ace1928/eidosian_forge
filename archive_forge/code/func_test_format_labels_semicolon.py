import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
def test_format_labels_semicolon(self):
    la = utils.format_labels(['K1=V1;K2=V2;K3=V3;K4=V4;K5=V5'])
    self.assertEqual({'K1': 'V1', 'K2': 'V2', 'K3': 'V3', 'K4': 'V4', 'K5': 'V5'}, la)