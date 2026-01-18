from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def is_too_young_object(obj, max_creation_timestamp):
    return is_created_after(obj['metadata']['creationTimestamp'], max_creation_timestamp)