from oslo_serialization import jsonutils
from keystone import exception
from keystone.i18n import _
@classmethod
def append_resource(cls, rel, data):
    cls.__resources[rel] = data
    cls.__serialized_resource_data = None