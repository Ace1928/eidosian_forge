from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def analyze_refs_from_pods(self, pods):
    for pod in pods:
        too_young = is_too_young_object(pod, self.max_creationTimestamp)
        if pod['status']['phase'] not in ('Running', 'Pending') and too_young:
            continue
        referrer = {'kind': pod['kind'], 'namespace': pod['metadata']['namespace'], 'name': pod['metadata']['name']}
        err = self.analyze_refs_from_pod_spec(pod['spec'], referrer)
        if err:
            return err
    return None