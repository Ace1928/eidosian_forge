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
def get_valid_type_create_data_2_24():
    public = [True, False]
    dhss = [True, False]
    snapshot = [None]
    create_from_snapshot = [None]
    extra_specs = [None, {'replication_type': 'writable', 'foo': 'bar'}]
    snapshot_none_combos = list(itertools.product(public, dhss, snapshot, create_from_snapshot, extra_specs))
    public = [True, False]
    dhss = [True, False]
    snapshot = [True]
    create_from_snapshot = [True, False, None]
    extra_specs = [None, {'replication_type': 'readable', 'foo': 'bar'}]
    snapshot_true_combos = list(itertools.product(public, dhss, snapshot, create_from_snapshot, extra_specs))
    public = [True, False]
    dhss = [True, False]
    snapshot = [False]
    create_from_snapshot = [False, None]
    extra_specs = [None, {'replication_type': 'dr', 'foo': 'bar'}]
    snapshot_false_combos = list(itertools.product(public, dhss, snapshot, create_from_snapshot, extra_specs))
    return snapshot_none_combos + snapshot_true_combos + snapshot_false_combos