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
def create_one_resource_type(attrs=None):
    """Create a fake MetadefResourceType member.

    :param attrs: A dictionary with all attributes of
        metadef_resource_type member
    :type attrs: dict
    :return: a fake MetadefResourceType object
    :rtype: A `metadef_resource_type.MetadefResourceType`
    """
    attrs = attrs or {}
    metadef_resource_type_info = {'name': 'OS::Compute::Quota', 'properties_target': 'image'}
    metadef_resource_type_info.update(attrs)
    return metadef_resource_type.MetadefResourceType(**metadef_resource_type_info)