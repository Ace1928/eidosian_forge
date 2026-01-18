from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
def remove_bundle(self, policy_content_spec: messages.Message) -> messages.Message:
    doomed_bundle = getattr(self.args, ARG_LABEL_BUNDLE)
    if doomed_bundle is None:
        raise exceptions.SafetyError('No bundle name specified!')
    bundles = protos.additional_properties_to_dict(policy_content_spec.bundles)
    found = bundles.pop(doomed_bundle, None)
    if found is None:
        raise exceptions.NoSuchContentError('{} is not installed.  Check that the name of the bundle is correct.'.format(doomed_bundle))
    policy_content_spec.bundles = protos.set_additional_properties(self.bundle_message(), bundles)
    return policy_content_spec