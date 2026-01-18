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
def stage_image(self, image, *, filename=None, data=None):
    """Stage binary image data

        :param image: The value can be the ID of a image or a
            :class:`~openstack.image.v2.image.Image` instance.
        :param filename: Optional name of the file to read data from.
        :param data: Optional data to be uploaded as an image.

        :returns: The results of image creation
        :rtype: :class:`~openstack.image.v2.image.Image`
        """
    if filename and data:
        raise exceptions.SDKException('filename and data are mutually exclusive')
    image = self._get_resource(_image.Image, image)
    if 'queued' != image.status:
        raise exceptions.SDKException('Image stage is only possible for images in the queued state. Current state is {status}'.format(status=image.status))
    if filename:
        image.data = open(filename, 'rb')
    elif data:
        image.data = data
    image.stage(self)
    image.fetch(self)
    return image