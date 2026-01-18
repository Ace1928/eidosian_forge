from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import function
from heat.engine import properties
from heat.engine import resource
def rebuild_lc_properties(self, instance_id):
    server = self.client_plugin('nova').get_server(instance_id)
    instance_props = {self.IMAGE_ID: server.image['id'], self.INSTANCE_TYPE: server.flavor['id'], self.KEY_NAME: server.key_name, self.SECURITY_GROUPS: [sg['name'] for sg in server.security_groups]}
    lc_props = function.resolve(self.properties.data)
    for key, value in instance_props.items():
        lc_props.setdefault(key, value)
    return lc_props