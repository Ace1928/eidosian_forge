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
def ex_get_image_from_family(self, image_family, ex_project_list=None, ex_standard_projects=True):
    """
        Return an GCENodeImage object based on an image family name.

        :param  image_family: The name of the 'Image Family' to return the
                              latest image from.
        :type   image_family: ``str``

        :param  ex_project_list: The name of the project to list for images.
                                 Examples include: 'debian-cloud'.
        :type   ex_project_list: ``list`` of ``str``, or ``None``

        :param  ex_standard_projects: If true, check in standard projects if
                                      the image is not found.
        :type   ex_standard_projects: ``bool``

        :return:  GCENodeImage object based on provided information or
                  ResourceNotFoundError if the image family is not found.
        :rtype:   :class:`GCENodeImage` or raise ``ResourceNotFoundError``
        """

    def _try_image_family(image_family, project=None):
        request = '/global/images/family/%s' % image_family
        save_request_path = self.connection.request_path
        if project:
            new_request_path = save_request_path.replace(self.project, project)
            self.connection.request_path = new_request_path
        try:
            response = self.connection.request(request, method='GET')
            image = self._to_node_image(response.object)
        except ResourceNotFoundError:
            image = None
        finally:
            self.connection.request_path = save_request_path
        return image
    image = None
    if image_family.startswith('https://'):
        response = self.connection.request(image_family, method='GET')
        return self._to_node_image(response.object)
    if not ex_project_list:
        image = _try_image_family(image_family)
    else:
        for img_proj in ex_project_list:
            image = _try_image_family(image_family, project=img_proj)
            if image:
                break
    if not image and ex_standard_projects:
        for img_proj, short_list in self.IMAGE_PROJECTS.items():
            for short_name in short_list:
                if image_family.startswith(short_name):
                    image = _try_image_family(image_family, project=img_proj)
                    if image:
                        break
    if not image:
        raise ResourceNotFoundError("Could not find image for family '%s'" % image_family, None, None)
    return image