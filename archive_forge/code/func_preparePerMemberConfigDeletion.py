from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import base
def preparePerMemberConfigDeletion(self, args, mask, patch):
    membership = base.ParseMembership(args, prompt=True, autoselect=True, search=True)
    patch.membershipSpecs = self.hubclient.ToMembershipSpecs({membership: self.messages.MembershipFeatureSpec()})
    mask.append('membership_specs')