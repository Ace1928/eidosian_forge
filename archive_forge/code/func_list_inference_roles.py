from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def list_inference_roles(self):
    """List all rule inferences.

        Valid HTTP return codes:

            * 200: All inference rules are returned

        :param kwargs: attributes provided will be passed to the server.

        :returns: a list of inference rules.
        :rtype: list of :class:`keystoneclient.v3.roles.InferenceRule`

        """
    return super(InferenceRuleManager, self).list()