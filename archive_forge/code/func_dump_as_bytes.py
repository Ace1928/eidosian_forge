from oslo_serialization import jsonutils
from oslo_serialization.serializer.base_serializer import BaseSerializer
def dump_as_bytes(self, obj):
    return jsonutils.dump_as_bytes(obj, default=self._default, encoding=self._encoding)