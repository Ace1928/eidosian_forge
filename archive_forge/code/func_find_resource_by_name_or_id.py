from oslo_config import cfg
from saharaclient.api import base as sahara_base
from saharaclient import client as sahara_client
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
def find_resource_by_name_or_id(self, resource_name, value):
    """Return the ID for the specified name or identifier.

        :param resource_name: API name of entity
        :param value: ID or name of entity
        :returns: the id of the requested :value:
        :raises exception.EntityNotFound:
        :raises exception.PhysicalResourceNameAmbiguity:
        """
    try:
        entity = getattr(self.client(), resource_name)
        return entity.get(value).id
    except sahara_base.APIException:
        return self.find_resource_by_name(resource_name, value)