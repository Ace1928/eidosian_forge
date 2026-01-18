from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def analyze_reference_image(self, image, referrer):
    result, error = parse_docker_image_ref(image, self.module)
    if error:
        return error
    if not result['hostname'] or not result['namespace']:
        return None
    if not result['digest']:
        if result['tag'] == '':
            result['tag'] = 'latest'
        key = '%s/%s:%s' % (result['namespace'], result['name'], result['tag'])
        if key not in self.used_tags:
            self.used_tags[key] = []
        self.used_tags[key].append(referrer)
    else:
        key = '%s/%s@%s' % (result['namespace'], result['name'], result['digest'])
        if key not in self.used_images:
            self.used_images[key] = []
        self.used_images[key].append(referrer)