from __future__ import absolute_import, division, print_function
import errno
import json
import os
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.image_archive import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.auth import (
from ansible_collections.community.docker.plugins.module_utils._api.constants import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils._api.utils.build import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def archive_image(self, name, tag):
    """
        Archive an image to a .tar file. Called when archive_path is passed.

        :param name: Name/repository of the image
        :type name: str
        :param tag: Optional image tag; assumed to be "latest" if None
        :type tag: str | None
        """
    if not tag:
        tag = 'latest'
    if is_image_name_id(name):
        image = self.client.find_image_by_id(name, accept_missing_image=True)
        image_name = name
    else:
        image = self.client.find_image(name=name, tag=tag)
        image_name = '%s:%s' % (name, tag)
    if not image:
        self.log('archive image: image %s not found' % image_name)
        return
    image_id = image['Id']
    action = self.archived_image_action(self.client.module.debug, self.archive_path, image_name, image_id)
    if action:
        self.results['actions'].append(action)
    self.results['changed'] = action is not None
    if not self.check_mode and self.results['changed']:
        self.log('Getting archive of image %s' % image_name)
        try:
            saved_image = self.client._stream_raw_result(self.client._get(self.client._url('/images/{0}/get', image_name), stream=True), DEFAULT_DATA_CHUNK_SIZE, False)
        except Exception as exc:
            self.fail('Error getting image %s - %s' % (image_name, to_native(exc)))
        try:
            with open(self.archive_path, 'wb') as fd:
                for chunk in saved_image:
                    fd.write(chunk)
        except Exception as exc:
            self.fail('Error writing image archive %s - %s' % (self.archive_path, to_native(exc)))
    self.results['image'] = image