from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
@removals.remove(message='Use %s.list_inference_roles' % deprecation_msg, version='3.9.0', removal_version='4.0.0')
def list_role_inferences(self, **kwargs):
    return InferenceRuleManager(self.client).list_inference_roles()