from openstack.clustering.v1 import _async_resource
from openstack.common import metadata
from openstack import resource
from openstack import utils
def policy_update(self, session, policy_id, **params):
    data = {'policy_id': policy_id}
    data.update(params)
    body = {'policy_update': data}
    return self.action(session, body)