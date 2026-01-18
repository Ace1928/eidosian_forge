from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.export import util
from googlecloudsdk.core.console import console_io
def no_fleet_default_cfg_flag(include_no: bool=False):
    """Flag for unsetting fleet default configuration."""
    flag = '--{}fleet-default-member-config'.format('no-' if include_no else '')
    return base.Argument(flag, action='store_true', help='Removes the fleet default configuration for policy controller.\n      Memberships configured with the fleet default will maintain their current\n      configuration.\n\n          $ {} {}\n      '.format('{command}', flag))