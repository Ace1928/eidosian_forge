from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def get_live_resource_data(self):
    try:
        resource_data = self._show_resource()
        if not resource_data:
            raise AttributeError()
    except Exception as ex:
        if self.client_plugin().is_not_found(ex) or isinstance(ex, AttributeError):
            raise exception.EntityNotFound(entity='Resource', name=self.name)
        raise
    return resource_data