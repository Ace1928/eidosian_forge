from openstack import exceptions
from openstack import resource
from openstack import utils
def remove_router_from_agent(self, session, router):
    body = {'router_id': router}
    url = utils.urljoin(self.base_path, self.id, 'l3-routers', router)
    session.delete(url, json=body)