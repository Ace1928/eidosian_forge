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
def ex_list_project_images(self, ex_project=None, ex_include_deprecated=False):
    """
        Return a list of image objects for a project. If no project is
        specified, only a list of 'global' images is returned.

        :keyword  ex_project: Optional alternate project name.
        :type     ex_project: ``str``, ``list`` of ``str``, or ``None``

        :keyword  ex_include_deprecated: If True, even DEPRECATED images will
                                         be returned.
        :type     ex_include_deprecated: ``bool``

        :return:  List of GCENodeImage objects
        :rtype:   ``list`` of :class:`GCENodeImage`
        """
    list_images = []
    request = '/global/images'
    if ex_project is None:
        response = self.connection.paginated_request(request, method='GET')
        for img in response.get('items', []):
            if 'deprecated' not in img:
                list_images.append(self._to_node_image(img))
            elif ex_include_deprecated:
                list_images.append(self._to_node_image(img))
    else:
        list_images = []
        save_request_path = self.connection.request_path
        if isinstance(ex_project, str):
            ex_project = [ex_project]
        for proj in ex_project:
            new_request_path = save_request_path.replace(self.project, proj)
            self.connection.request_path = new_request_path
            try:
                response = self.connection.paginated_request(request, method='GET')
            except Exception:
                raise
            finally:
                self.connection.request_path = save_request_path
            for img in response.get('items', []):
                if 'deprecated' not in img:
                    list_images.append(self._to_node_image(img))
                elif ex_include_deprecated:
                    list_images.append(self._to_node_image(img))
    return list_images