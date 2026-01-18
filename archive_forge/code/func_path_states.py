from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util as fleet_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.core import exceptions as gcloud_exceptions
import six
def path_states(self, args: parser_extensions.Namespace) -> SpecMapping:
    """Retrieves membership states specified that exist in the Feature.

    Args:
      args: The argparse object passed to the command.

    Returns:
      A dict mapping a path to the membership spec.

    Raises:
      exceptions.DisabledMembershipError: If the membership is invalid or not
      enabled.
    """
    memberships_paths = self._membership_paths(args)
    states = {fleet_util.MembershipPartialName(path): (path, state) for path, state in self.current_states().items() if fleet_util.MembershipPartialName(path) in memberships_paths}
    msg = 'Policy Controller is not enabled for membership {}'
    missing_memberships = [exceptions.InvalidPocoMembershipError(msg.format(path)) for path in memberships_paths if path not in states]
    if missing_memberships:
        raise exceptions.InvalidPocoMembershipError(missing_memberships)
    return {path: spec for path, spec in states.values()}