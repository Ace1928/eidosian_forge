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
@utils.arg('volume_type', metavar='<volume_type>', help='Name or ID of the volume type.')
@utils.arg('project', metavar='<project_id>', help='ID of project for which to set default type.')
def do_default_type_set(cs, args):
    """Sets a default volume type for a project."""
    volume_type = args.volume_type
    project = args.project
    default_type = cs.default_types.create(volume_type, project)
    shell_utils.print_dict(default_type._info)