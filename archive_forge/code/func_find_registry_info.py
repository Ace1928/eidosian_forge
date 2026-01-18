from __future__ import (absolute_import, division, print_function)
import traceback
from urllib.parse import urlparse
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def find_registry_info(self):

    def _determine_registry(image_stream):
        public, internal = (None, None)
        docker_repo = image_stream['status'].get('publicDockerImageRepository')
        if docker_repo:
            ref, err = parse_docker_image_ref(docker_repo, self.module)
            public = ref['hostname']
        docker_repo = image_stream['status'].get('dockerImageRepository')
        if docker_repo:
            ref, err = parse_docker_image_ref(docker_repo, self.module)
            internal = ref['hostname']
        return (internal, public)
    for stream in self.list_image_streams(namespace='openshift'):
        internal, public = _determine_registry(stream)
        if not public and (not internal):
            self.fail_json(msg='The integrated registry has not been configured')
        return (internal, public)
    for stream in self.list_image_streams():
        internal, public = _determine_registry(stream)
        if not public and (not internal):
            self.fail_json(msg='The integrated registry has not been configured')
        return (internal, public)
    self.fail_json(msg='No Image Streams could be located to retrieve registry info.')