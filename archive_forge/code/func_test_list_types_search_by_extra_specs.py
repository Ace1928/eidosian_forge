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
def test_list_types_search_by_extra_specs(self):
    search_opts = {'extra_specs': {'aa': 'bb'}}
    cs.share_types.list(search_opts=search_opts)
    expect = '/types?extra_specs=%7B%27aa%27%3A+%27bb%27%7D&is_public=all'
    cs.assert_called('GET', expect)