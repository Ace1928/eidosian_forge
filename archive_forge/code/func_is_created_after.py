from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def is_created_after(creation_timestamp, max_creation_timestamp):
    if not max_creation_timestamp:
        return False
    creationTimestamp = datetime.strptime(creation_timestamp, '%Y-%m-%dT%H:%M:%SZ')
    return creationTimestamp > max_creation_timestamp