from __future__ import (absolute_import, division, print_function)
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def follow_imagestream_tag_reference(stream, tag):
    multiple = False

    def _imagestream_has_tag():
        for ref in stream['spec'].get('tags', []):
            if ref['name'] == tag:
                return ref
        return None

    def _imagestream_split_tag(name):
        parts = name.split(':')
        name = parts[0]
        tag = ''
        if len(parts) > 1:
            tag = parts[1]
        if len(tag) == 0:
            tag = 'latest'
        return (name, tag, len(parts) == 2)
    content = []
    err_cross_stream_ref = 'tag %s points to an imagestreamtag from another ImageStream' % tag
    while True:
        if tag in content:
            return (tag, None, multiple, 'tag %s on the image stream is a reference to same tag' % tag)
        content.append(tag)
        tag_ref = _imagestream_has_tag()
        if not tag_ref:
            return (None, None, multiple, err_stream_not_found_ref)
        if not tag_ref.get('from') or tag_ref['from']['kind'] != 'ImageStreamTag':
            return (tag, tag_ref, multiple, None)
        if tag_ref['from']['namespace'] != '' and tag_ref['from']['namespace'] != stream['metadata']['namespace']:
            return (tag, None, multiple, err_cross_stream_ref)
        if ':' in tag_ref['from']['name']:
            name, tagref, result = _imagestream_split_tag(tag_ref['from']['name'])
            if not result:
                return (tag, None, multiple, 'tag %s points to an invalid imagestreamtag' % tag)
            if name != stream['metadata']['namespace']:
                return (tag, None, multiple, err_cross_stream_ref)
            tag = tagref
        else:
            tag = tag_ref['from']['name']
        multiple = True