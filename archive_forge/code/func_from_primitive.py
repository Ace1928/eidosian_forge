from oslo_serialization import jsonutils as json
from oslo_versionedobjects import fields
def from_primitive(self, obj, attr, value):
    return self.coerce(obj, attr, value)