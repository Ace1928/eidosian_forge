import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_create_image(self, name, volume, description=None, family=None, guest_os_features=None, use_existing=True, wait_for_completion=True, ex_licenses=None, ex_labels=None):
    """
        Create an image from the provided volume.

        :param  name: The name of the image to create.
        :type   name: ``str``

        :param  volume: The volume to use to create the image, or the
                        Google Cloud Storage URI
        :type   volume: ``str`` or :class:`StorageVolume`

        :keyword  description: Description of the new Image
        :type     description: ``str``

        :keyword  family: The name of the image family to which this image
                          belongs. If you create resources by specifying an
                          image family instead of a specific image name, the
                          resource uses the latest non-deprecated image that
                          is set with that family name.
        :type     family: ``str``

        :keyword  guest_os_features: Features of the guest operating system,
                                     valid for bootable images only.
        :type     guest_os_features: ``list`` of ``str`` or ``None``

        :keyword  ex_licenses: List of strings representing licenses
                               to be associated with the image.
        :type     ex_licenses: ``list`` of ``str``

        :keyword  ex_labels: Labels dictionary for image.
        :type     ex_labels: ``dict`` or ``None``

        :keyword  use_existing: If True and an image with the given name
                                already exists, return an object for that
                                image instead of attempting to create
                                a new image.
        :type     use_existing: ``bool``

        :keyword  wait_for_completion: If True, wait until the new image is
                                       created before returning a new NodeImage
                                       Otherwise, return a new NodeImage
                                       instance, and let the user track the
                                       creation progress
        :type     wait_for_completion: ``bool``

        :return:  A GCENodeImage object for the new image
        :rtype:   :class:`GCENodeImage`

        """
    image_data = {}
    image_data['name'] = name
    image_data['description'] = description
    image_data['family'] = family
    if isinstance(volume, StorageVolume):
        image_data['sourceDisk'] = volume.extra['selfLink']
        image_data['zone'] = volume.extra['zone'].name
    elif isinstance(volume, str) and volume.startswith('https://') and volume.endswith('tar.gz'):
        image_data['rawDisk'] = {'source': volume, 'containerType': 'TAR'}
    else:
        raise ValueError('Source must be instance of StorageVolume or URI')
    if ex_licenses:
        if isinstance(ex_licenses, str):
            ex_licenses = [ex_licenses]
        image_data['licenses'] = ex_licenses
    if ex_labels:
        image_data['labels'] = ex_labels
    if guest_os_features:
        image_data['guestOsFeatures'] = []
        if isinstance(guest_os_features, str):
            guest_os_features = [guest_os_features]
        for feature in guest_os_features:
            image_data['guestOsFeatures'].append({'type': feature})
    request = '/global/images'
    try:
        if wait_for_completion:
            self.connection.async_request(request, method='POST', data=image_data)
        else:
            self.connection.request(request, method='POST', data=image_data)
    except ResourceExistsError as e:
        if not use_existing:
            raise e
    return self.ex_get_image(name)