from oslo_serialization import jsonutils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from urllib import parse
def physical_resource_name(self):
    name = self.properties[self.NAME]
    if name is not None:
        return name
    return super(ZaqarQueue, self).physical_resource_name()