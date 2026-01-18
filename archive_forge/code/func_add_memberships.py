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
def add_memberships(self):
    """Adds handling for single, multiple or all memberships."""
    group = self.parser.add_argument_group('Membership flags.', mutex=True)
    resources.AddMembershipResourceArg(group, plural=True, membership_help='The membership names to act on, separated by commas if multiple are supplied. Ignored if --all-memberships is supplied; if neither is supplied, a prompt will appear with all available memberships.')
    group.add_argument('--all-memberships', action='store_true', help='If supplied, apply to all Policy Controllers memberships in the fleet.', default=False)