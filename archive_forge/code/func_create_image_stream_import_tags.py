from __future__ import (absolute_import, division, print_function)
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def create_image_stream_import_tags(self, stream, tags):
    isi, streamInsecure = self.create_image_stream_import(stream)
    for k in tags:
        insecure = streamInsecure
        scheduled = self.params.get('scheduled')
        old_tag = None
        for t in stream.get('spec', {}).get('tags', []):
            if t['name'] == k:
                old_tag = t
                break
        if old_tag:
            insecure = insecure or old_tag['importPolicy'].get('insecure')
            scheduled = scheduled or old_tag['importPolicy'].get('scheduled')
        images = isi['spec'].get('images', [])
        images.append({'from': {'kind': 'DockerImage', 'name': tags.get(k)}, 'to': {'name': k}, 'importPolicy': {'insecure': insecure, 'scheduled': scheduled}, 'referencePolicy': self.ref_policy})
        isi['spec']['images'] = images
    return isi