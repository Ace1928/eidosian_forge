from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.image_archive import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.constants import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def get_export_reason(self):
    if self.force:
        return 'Exporting since force=true'
    try:
        archived_images = load_archived_image_manifest(self.path)
        if archived_images is None:
            return 'Overwriting since no image is present in archive'
    except ImageArchiveInvalidException as exc:
        self.log('Unable to extract manifest summary from archive: %s' % to_native(exc))
        return 'Overwriting an unreadable archive file'
    left_names = list(self.names)
    for archived_image in archived_images:
        found = False
        for i, name in enumerate(left_names):
            if name['id'] == api_image_id(archived_image.image_id) and [name['joined']] == archived_image.repo_tags:
                del left_names[i]
                found = True
                break
        if not found:
            return 'Overwriting archive since it contains unexpected image %s named %s' % (archived_image.image_id, ', '.join(archived_image.repo_tags))
    if left_names:
        return 'Overwriting archive since it is missing image(s) %s' % ', '.join([name['joined'] for name in left_names])
    return None