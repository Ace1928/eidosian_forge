import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
def get_cloud(log_file):
    CLOUD = {'clouds': {'megacloud': {'cloud': 'megadodo', 'auth': {'project_name': 'heart-o-gold', 'username': 'zaphod'}, 'region_name': 'occ-cloud,krikkit,occ-env', 'log_file': log_file, 'log_level': 'debug', 'cert': 'mycert', 'key': 'mickey'}}}
    return CLOUD