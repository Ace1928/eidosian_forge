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
def set_default_cfg(self, feature: messages.Message, membership: messages.Message) -> messages.Message:
    """Sets membership to the default fleet configuration.

    Args:
      feature: The feature spec for the project.
      membership: The membership spec being updated.

    Returns:
      Updated MembershipFeatureSpec.
    Raises:
      MissingFleetDefaultMemberConfig: If none exists on the feature.
    """
    if feature.fleetDefaultMemberConfig is None:
        project = feature.name.split('/')[1]
        msg = "No fleet default member config specified for project {}. Use '... enable --fleet-default-member-config=config.yaml'."
        raise exceptions.MissingFleetDefaultMemberConfig(msg.format(project))
    self.set_origin_fleet(membership)
    membership.policycontroller = feature.fleetDefaultMemberConfig.policycontroller