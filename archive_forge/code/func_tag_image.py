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
def tag_image(self, name, tag, repository, push=False):
    """
        Tag an image into a repository.

        :param name: name of the image. required.
        :param tag: image tag.
        :param repository: path to the repository. required.
        :param push: bool. push the image once it's tagged.
        :return: None
        """
    repo, repo_tag = parse_repository_tag(repository)
    if not repo_tag:
        repo_tag = 'latest'
        if tag:
            repo_tag = tag
    image = self.client.find_image(name=repo, tag=repo_tag)
    found = 'found' if image else 'not found'
    self.log('image %s was %s' % (repo, found))
    if not image or self.force_tag:
        image_name = name
        if not is_image_name_id(name) and tag and (not name.endswith(':' + tag)):
            image_name = '%s:%s' % (name, tag)
        self.log('tagging %s to %s:%s' % (image_name, repo, repo_tag))
        self.results['changed'] = True
        self.results['actions'].append('Tagged image %s to %s:%s' % (image_name, repo, repo_tag))
        if not self.check_mode:
            try:
                params = {'tag': repo_tag, 'repo': repo, 'force': True}
                res = self.client._post(self.client._url('/images/{0}/tag', image_name), params=params)
                self.client._raise_for_status(res)
                if res.status_code != 201:
                    raise Exception('Tag operation failed.')
            except Exception as exc:
                self.fail('Error: failed to tag image - %s' % to_native(exc))
            self.results['image'] = self.client.find_image(name=repo, tag=repo_tag)
            if image and image['Id'] == self.results['image']['Id']:
                self.results['changed'] = False
    if push:
        self.push_image(repo, repo_tag)