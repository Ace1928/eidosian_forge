from oslo_config import cfg
from saharaclient.api import base as sahara_base
from saharaclient import client as sahara_client
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
def find_resource_by_name(self, resource_name, value):
    """Return the ID for the specified entity name.

        :raises exception.EntityNotFound:
        :raises exception.PhysicalResourceNameAmbiguity:
        """
    try:
        filters = {'name': value}
        obj = getattr(self.client(), resource_name)
        obj_list = obj.find(**filters)
    except sahara_base.APIException as ex:
        raise exception.Error(_('Error retrieving %(entity)s list from sahara: %(err)s') % dict(entity=resource_name, err=str(ex)))
    num_matches = len(obj_list)
    if num_matches == 0:
        raise exception.EntityNotFound(entity=resource_name or 'entity', name=value)
    elif num_matches > 1:
        raise exception.PhysicalResourceNameAmbiguity(name=value)
    else:
        return obj_list[0].id