from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def assertRoute(self, mapper, path, method, action, controller, params=None):
    params = params or {}
    route = mapper.match(path, {'REQUEST_METHOD': method})
    self.assertIsNotNone(route)
    self.assertEqual(action, route['action'])
    class_name = reflection.get_class_name(route['controller'].controller, fully_qualified=False)
    self.assertEqual(controller, class_name)
    del route['action']
    del route['controller']
    self.assertEqual(params, route)