from __future__ import (absolute_import, division, print_function)
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def create_image_import(self, ref):
    kind = 'ImageStream'
    api_version = 'image.openshift.io/v1'
    params = dict(kind=kind, api_version=api_version, name=ref.get('name'), namespace=self.params.get('namespace'))
    result = self.kubernetes_facts(**params)
    if not result['api_found']:
        msg = ('Failed to find API for resource with apiVersion "{0}" and kind "{1}"'.format(api_version, kind),)
        self.fail_json(msg=msg)
    imagestream = None
    if len(result['resources']) > 0:
        imagestream = result['resources'][0]
    stream, isi = (None, None)
    if not imagestream:
        stream, isi = self.create_image_stream(ref)
    elif self.params.get('all') and (not ref['tag']):
        stream, isi = self.import_all(imagestream)
    else:
        stream, isi = self.import_tag(imagestream, ref['tag'])
    return isi