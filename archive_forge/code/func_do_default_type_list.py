import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.62')
@utils.arg('--project-id', metavar='<project_id>', default=None, help='ID of project for which to show the default type.')
def do_default_type_list(cs, args):
    """Lists all default volume types."""
    project_id = args.project_id
    default_types = cs.default_types.list(project_id)
    columns = ['Volume Type ID', 'Project ID']
    if project_id:
        shell_utils.print_dict(default_types._info)
    else:
        shell_utils.print_list(default_types, columns)