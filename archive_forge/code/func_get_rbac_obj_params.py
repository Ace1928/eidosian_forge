from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def get_rbac_obj_params(client, obj_type, obj_id_or_name):
    resource, cmd_resource = _get_cmd_resource(obj_type)
    obj_id = neutronV20.find_resourceid_by_name_or_id(client=client, resource=resource, name_or_id=obj_id_or_name, cmd_resource=cmd_resource)
    return (obj_id, cmd_resource)