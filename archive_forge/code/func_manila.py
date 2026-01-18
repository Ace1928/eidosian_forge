import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def manila(self, action, flags='', params='', fail_ok=False, endpoint_type='publicURL', merge_stderr=False, microversion=None):
    """Executes manila command for the given action.

        :param action: the cli command to run using manila
        :type action: string
        :param flags: any optional cli flags to use. For specifying
                      microversion, please, use 'microversion' param
        :type flags: string
        :param params: any optional positional args to use
        :type params: string
        :param fail_ok: if True an exception is not raised when the
                        cli return code is non-zero
        :type fail_ok: boolean
        :param endpoint_type: the type of endpoint for the service
        :type endpoint_type: string
        :param merge_stderr: if True the stderr buffer is merged into stdout
        :type merge_stderr: boolean
        :param microversion: API microversion to be used for request
        :type microversion: str
        """
    flags += ' --endpoint-type %s' % endpoint_type
    if not microversion:
        microversion = CONF.max_api_microversion
    flags += ' --os-share-api-version %s' % microversion
    return self.cmd_with_auth('manila', action, flags, params, fail_ok, merge_stderr)