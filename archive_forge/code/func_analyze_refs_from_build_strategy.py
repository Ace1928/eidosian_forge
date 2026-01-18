from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def analyze_refs_from_build_strategy(self, resources):
    keys = ('BuildConfig', 'Build')
    for k, objects in iteritems(resources):
        if k not in keys:
            continue
        for obj in objects:
            referrer = {'kind': obj['kind'], 'namespace': obj['metadata']['namespace'], 'name': obj['metadata']['name']}
            error = self.analyze_refs_from_strategy(obj['spec']['strategy'], obj['metadata']['namespace'], referrer)
            if error is not None:
                return '%s/%s/%s: %s' % (referrer['kind'], referrer['namespace'], referrer['name'], error)