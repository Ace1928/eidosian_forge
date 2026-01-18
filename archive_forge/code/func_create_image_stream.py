from __future__ import (absolute_import, division, print_function)
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def create_image_stream(self, ref):
    """
            Create new ImageStream and accompanying ImageStreamImport
        """
    source = self.params.get('source')
    if not source:
        source = ref['source']
    stream = dict(apiVersion='image.openshift.io/v1', kind='ImageStream', metadata=dict(name=ref['name'], namespace=self.params.get('namespace')))
    if self.params.get('all') and (not ref['tag']):
        spec = dict(dockerImageRepository=source)
        isi = self.create_image_stream_import_all(stream, source)
    else:
        spec = dict(tags=[{'from': {'kind': 'DockerImage', 'name': source}, 'referencePolicy': self.ref_policy}])
        tags = {ref['tag']: source}
        isi = self.create_image_stream_import_tags(stream, tags)
    stream.update(dict(spec=spec))
    return (stream, isi)