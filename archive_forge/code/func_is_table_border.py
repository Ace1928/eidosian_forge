import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def is_table_border(t):
    return str(t).startswith('+')