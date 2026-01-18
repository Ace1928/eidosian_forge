import os
import time
import typing as ty
import warnings
from openstack import exceptions
from openstack.image.v2 import cache as _cache
from openstack.image.v2 import image as _image
from openstack.image.v2 import member as _member
from openstack.image.v2 import metadef_namespace as _metadef_namespace
from openstack.image.v2 import metadef_object as _metadef_object
from openstack.image.v2 import metadef_property as _metadef_property
from openstack.image.v2 import metadef_resource_type as _metadef_resource_type
from openstack.image.v2 import metadef_schema as _metadef_schema
from openstack.image.v2 import schema as _schema
from openstack.image.v2 import service_info as _si
from openstack.image.v2 import task as _task
from openstack import proxy
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
def create_metadef_object(self, namespace, **attrs):
    """Create a new object from namespace

        :param namespace: The value can be either the name of a metadef
            namespace or a
            :class:`~openstack.image.v2.metadef_namespace.MetadefNamespace`
            instance.
        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.image.v2.metadef_object.MetadefObject`,
            comprised of the properties on the Metadef object class.

        :returns: A metadef namespace
        :rtype: :class:`~openstack.image.v2.metadef_object.MetadefObject`
        """
    namespace_name = resource.Resource._get_id(namespace)
    return self._create(_metadef_object.MetadefObject, namespace_name=namespace_name, **attrs)