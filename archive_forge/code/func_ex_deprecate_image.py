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
def ex_deprecate_image(self, image, replacement, state=None, deprecated=None, obsolete=None, deleted=None):
    """
        Deprecate a specific image resource.

        :param  image: Image object to deprecate
        :type   image: ``str`` or :class: `GCENodeImage`

        :param  replacement: Image object to use as a replacement
        :type   replacement: ``str`` or :class: `GCENodeImage`

        :param  state: State of the image
        :type   state: ``str``

        :param  deprecated: RFC3339 timestamp to mark DEPRECATED
        :type   deprecated: ``str`` or ``None``

        :param  obsolete: RFC3339 timestamp to mark OBSOLETE
        :type   obsolete: ``str`` or ``None``

        :param  deleted: RFC3339 timestamp to mark DELETED
        :type   deleted: ``str`` or ``None``

        :return: True if successful
        :rtype:  ``bool``
        """
    if not hasattr(image, 'name'):
        image = self.ex_get_image(image)
    if not hasattr(replacement, 'name'):
        replacement = self.ex_get_image(replacement)
    if state is None:
        state = 'DEPRECATED'
    possible_states = ['ACTIVE', 'DELETED', 'DEPRECATED', 'OBSOLETE']
    if state not in possible_states:
        raise ValueError('state must be one of %s' % ','.join(possible_states))
    if state == 'ACTIVE':
        image_data = {}
    else:
        image_data = {'state': state, 'replacement': replacement.extra['selfLink']}
        for attribute, value in [('deprecated', deprecated), ('obsolete', obsolete), ('deleted', deleted)]:
            if value is None:
                continue
            try:
                timestamp_to_datetime(value)
            except Exception:
                raise ValueError('%s must be an RFC3339 timestamp' % attribute)
            image_data[attribute] = value
    request = '/global/images/%s/deprecate' % image.name
    self.connection.request(request, method='POST', data=image_data).object
    return True