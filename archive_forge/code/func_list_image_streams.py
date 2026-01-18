from __future__ import (absolute_import, division, print_function)
import traceback
from urllib.parse import urlparse
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def list_image_streams(self, namespace=None):
    kind = 'ImageStream'
    api_version = 'image.openshift.io/v1'
    params = dict(kind=kind, api_version=api_version, namespace=namespace)
    result = self.kubernetes_facts(**params)
    imagestream = []
    if len(result['resources']) > 0:
        imagestream = result['resources']
    return imagestream