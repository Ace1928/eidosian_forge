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
def list_share_types(self, list_all=True, columns=None, search_opts=None, microversion=None):
    """List share types.

        :param list_all: bool -- whether to list all share types or only public
        :param search_opts: dict search_opts for filter search.
        :param columns: comma separated string of columns.
            Example, "--columns id,name"
        """
    cmd = 'type-list'
    if list_all:
        cmd += ' --all'
    if search_opts is not None:
        extra_specs = search_opts.get('extra_specs')
        if extra_specs:
            cmd += ' --extra_specs'
            for spec_key in extra_specs.keys():
                cmd += ' ' + spec_key + '=' + extra_specs[spec_key]
    if columns is not None:
        cmd += ' --columns ' + columns
    share_types_raw = self.manila(cmd, microversion=microversion)
    share_types = output_parser.listing(share_types_raw)
    return share_types