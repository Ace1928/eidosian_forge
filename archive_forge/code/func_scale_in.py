from openstack.clustering.v1 import _async_resource
from openstack.common import metadata
from openstack import resource
from openstack import utils
def scale_in(self, session, count=None):
    body = {'scale_in': {'count': count}}
    return self.action(session, body)