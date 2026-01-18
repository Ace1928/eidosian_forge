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
def create_one_import_info(attrs=None):
    """Create a fake import info.

    :param attrs: A dictionary with all attributes of import info
    :type attrs: dict
    :return: A fake Import object.
    :rtype: `openstack.image.v2.service_info.Import`
    """
    attrs = attrs or {}
    import_info = {'import-methods': {'description': 'Import methods available.', 'type': 'array', 'value': ['glance-direct', 'web-download', 'glance-download', 'copy-image']}}
    import_info.update(attrs)
    return _service_info.Import(**import_info)