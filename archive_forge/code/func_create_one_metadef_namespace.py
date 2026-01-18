import random
from unittest import mock
import uuid
from openstack.image.v2 import _proxy
from openstack.image.v2 import cache
from openstack.image.v2 import image
from openstack.image.v2 import member
from openstack.image.v2 import metadef_namespace
from openstack.image.v2 import metadef_object
from openstack.image.v2 import metadef_property
from openstack.image.v2 import metadef_resource_type
from openstack.image.v2 import service_info as _service_info
from openstack.image.v2 import task
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def create_one_metadef_namespace(attrs=None):
    """Create a fake MetadefNamespace member.

    :param attrs: A dictionary with all attributes of metadef_namespace member
    :type attrs: dict
    :return: a list of MetadefNamespace objects
    :rtype: list of `metadef_namespace.MetadefNamespace`
    """
    attrs = attrs or {}
    metadef_namespace_list = {'created_at': '2022-08-17T11:30:22Z', 'display_name': 'Flavor Quota', 'namespace': 'OS::Compute::Quota', 'owner': 'admin', 'visibility': 'public'}
    metadef_namespace_list.update(attrs)
    return metadef_namespace.MetadefNamespace(**metadef_namespace_list)