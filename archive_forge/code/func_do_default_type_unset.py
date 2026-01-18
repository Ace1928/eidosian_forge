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
@utils.arg('project_id', metavar='<project_id>', nargs='+', help='ID of project for which to unset default type.')
def do_default_type_unset(cs, args):
    """Unset default volume types."""
    for project_id in args.project_id:
        try:
            cs.default_types.delete(project_id)
            print('Default volume type for project %s has been unset successfully.' % project_id)
        except Exception as e:
            print('Unset for default volume type for project %s failed: %s' % (project_id, e))