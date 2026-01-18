from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def tag_images(self):
    if is_image_name_id(self.name):
        image = self.client.find_image_by_id(self.name, accept_missing_image=False)
    else:
        image = self.client.find_image(name=self.name, tag=self.tag)
        if not image:
            self.fail('Cannot find image %s:%s' % (self.name, self.tag))
    before = []
    after = []
    tagged_images = []
    results = dict(changed=False, actions=[], image=image, tagged_images=tagged_images, diff=dict(before=dict(images=before), after=dict(images=after)))
    for repository, tag in self.repositories:
        tagged, msg, old_image = self.tag_image(image, repository, tag)
        before.append(image_info(repository, tag, old_image))
        after.append(image_info(repository, tag, image if tagged else old_image))
        if tagged:
            results['changed'] = True
            results['actions'].append('Tagged image %s as %s:%s: %s' % (image['Id'], repository, tag, msg))
            tagged_images.append('%s:%s' % (repository, tag))
        else:
            results['actions'].append('Not tagged image %s as %s:%s: %s' % (image['Id'], repository, tag, msg))
    return results