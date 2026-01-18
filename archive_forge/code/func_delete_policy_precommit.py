from oslo_log import log as logging
from neutron_lib.api import validators as lib_validators
from neutron_lib.callbacks import events
from neutron_lib.callbacks import registry
from neutron_lib.services.qos import constants
def delete_policy_precommit(self, context, policy):
    """Delete policy precommit.

        This method can be implemented by the specific driver subclass
        to handle delete precommit event of a policy that is being deleted.

        :param context: current running context information
        :param policy: a QoSPolicy object being deleted
        """