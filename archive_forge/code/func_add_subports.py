from openstack.common import tag
from openstack import exceptions
from openstack import resource
from openstack import utils
def add_subports(self, session, subports):
    url = utils.urljoin('/trunks', self.id, 'add_subports')
    resp = session.put(url, json={'sub_ports': subports})
    exceptions.raise_from_response(resp)
    self._body.attributes.update(resp.json())
    return self