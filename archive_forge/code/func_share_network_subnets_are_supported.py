import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def share_network_subnets_are_supported():
    return is_microversion_supported('2.51')