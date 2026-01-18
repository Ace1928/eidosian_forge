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
def delete_metadef_resource_type_association(self, metadef_resource_type, metadef_namespace, ignore_missing=True):
    """Removes a resource type association in a namespace.

        :param metadef_resource_type: The value can be either the name of
            a metadef resource type association or an
            :class:`~openstack.image.v2.metadef_resource_type.MetadefResourceTypeAssociation`
            instance.
        :param metadef_namespace: The value can be either the name of metadef
            namespace or an
            :class:`~openstack.image.v2.metadef_namespace.MetadefNamespace`
            instance
        :param bool ignore_missing: When set to ``False``,
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the metadef resource type association does not exist.
        :returns: ``None``
        """
    namespace_name = resource.Resource._get_id(metadef_namespace)
    self._delete(_metadef_resource_type.MetadefResourceTypeAssociation, metadef_resource_type, namespace_name=namespace_name, ignore_missing=ignore_missing)