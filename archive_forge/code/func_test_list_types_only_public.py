import copy
import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_types
def test_list_types_only_public(self):
    cs.share_types.list(show_all=False)
    cs.assert_called('GET', '/types')