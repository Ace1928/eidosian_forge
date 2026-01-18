from openstack.clustering.v1 import _async_resource
from openstack.common import metadata
from openstack import resource
from openstack import utils
def policy_detach(self, session, policy_id):
    body = {'policy_detach': {'policy_id': policy_id}}
    return self.action(session, body)