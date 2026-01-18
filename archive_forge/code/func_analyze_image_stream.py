from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def analyze_image_stream(self, resources):
    error = self.analyze_refs_from_pods(resources['Pod'])
    if error:
        return (None, None, error)
    error = self.analyze_refs_pod_creators(resources)
    if error:
        return (None, None, error)
    error = self.analyze_refs_from_build_strategy(resources)
    return (self.used_tags, self.used_images, error)