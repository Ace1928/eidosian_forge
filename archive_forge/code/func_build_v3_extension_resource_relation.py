from oslo_serialization import jsonutils
from keystone import exception
from keystone.i18n import _
def build_v3_extension_resource_relation(extension_name, extension_version, resource_name):
    return 'https://docs.openstack.org/api/openstack-identity/3/ext/%s/%s/rel/%s' % (extension_name, extension_version, resource_name)