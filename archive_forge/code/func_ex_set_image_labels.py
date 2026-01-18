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
def ex_set_image_labels(self, image, labels):
    """
        Set labels for the specified image.

        :keyword  image: The existing target Image for the request.
        :type     image: ``NodeImage``

        :keyword  labels: Set (or clear with None) labels for this image.
        :type     labels: ``dict`` or ``None``

        :return: True if successful
        :rtype:  ``bool``
        """
    if not isinstance(image, NodeImage):
        raise ValueError('Must specify a valid libcloud image object.')
    current_fp = image.extra['labelFingerprint']
    body = {'labels': labels, 'labelFingerprint': current_fp}
    request = '/global/images/%s/setLabels' % image.name
    self.connection.async_request(request, method='POST', data=body)
    return True