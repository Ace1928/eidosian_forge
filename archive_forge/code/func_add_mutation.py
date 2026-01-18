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
def add_mutation(self):
    """Adds handling for mutation enablement."""
    group = self.parser.add_group('Mutation flags.', mutex=True)
    group.add_argument('--no-mutation', action='store_true', help='Disables mutation support.')
    group.add_argument('--mutation', action='store_true', help='If set, enable support for mutation. (To disable, use --no-mutation)')