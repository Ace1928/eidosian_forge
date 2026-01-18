from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.image_archive import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.constants import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def export_images(self):
    image_names = [name['joined'] for name in self.names]
    image_names_str = ', '.join(image_names)
    if len(image_names) == 1:
        self.log('Getting archive of image %s' % image_names[0])
        try:
            chunks = self.client._stream_raw_result(self.client._get(self.client._url('/images/{0}/get', image_names[0]), stream=True), DEFAULT_DATA_CHUNK_SIZE, False)
        except Exception as exc:
            self.fail('Error getting image %s - %s' % (image_names[0], to_native(exc)))
    else:
        self.log('Getting archive of images %s' % image_names_str)
        try:
            chunks = self.client._stream_raw_result(self.client._get(self.client._url('/images/get'), stream=True, params={'names': image_names}), DEFAULT_DATA_CHUNK_SIZE, False)
        except Exception as exc:
            self.fail('Error getting images %s - %s' % (image_names_str, to_native(exc)))
    self.write_chunks(chunks)