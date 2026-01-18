from oslo_serialization import jsonutils
from keystone import exception
from keystone.i18n import _
def build_v3_parameter_relation(parameter_name):
    return 'https://docs.openstack.org/api/openstack-identity/3/param/%s' % parameter_name