import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
def boolenv(*vars, default=False):
    """Search for the first defined of possibly many bool-like env vars.

    Returns the first environment variable defined in vars, or returns the
    default.

    :param vars: Arbitrary strings to search for. Case sensitive.
    :param default: The default to return if no value found.
    :returns: A boolean corresponding to the value found, else the default if
        no value found.
    """
    for v in vars:
        value = os.environ.get(v, None)
        if value:
            return bool_from_str(value)
    return default