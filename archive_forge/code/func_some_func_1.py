from unittest import mock
import ddt
import manilaclient
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient.tests.unit import utils
@api_versions.wraps('2.2', '2.6')
@cliutils.arg('name_1', help='Name of the something')
@cliutils.arg('action_1', help='Some action')
def some_func_1(cs, args):
    pass