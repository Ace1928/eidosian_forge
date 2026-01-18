from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_images_common import (
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def prune_image_streams(self, stream):
    name = stream['metadata']['namespace'] + '/' + stream['metadata']['name']
    if is_too_young_object(stream, self.max_creation_timestamp):
        return (None, [])
    facts = self.kubernetes_facts(kind='ImageStream', api_version=ApiConfiguration.get('ImageStream'), name=stream['metadata']['name'], namespace=stream['metadata']['namespace'])
    image_stream = facts.get('resources')
    if len(image_stream) != 1:
        return (None, [])
    stream = image_stream[0]
    namespace = self.params.get('namespace')
    stream_to_update = not namespace or stream['metadata']['namespace'] == namespace
    manifests_to_delete, images_to_delete = ([], [])
    deleted_items = False
    if stream_to_update:
        tags = stream['status'].get('tags', [])
        for idx, tag_event_list in enumerate(tags):
            filtered_tag_event, tag_manifests_to_delete, tag_images_to_delete = self.prune_image_stream_tag(stream, tag_event_list)
            stream['status']['tags'][idx]['items'] = filtered_tag_event
            manifests_to_delete += tag_manifests_to_delete
            images_to_delete += tag_images_to_delete
            deleted_items = deleted_items or len(tag_images_to_delete) > 0
    tags = []
    for tag in stream['status'].get('tags', []):
        if tag['items'] is None or len(tag['items']) == 0:
            continue
        tags.append(tag)
    stream['status']['tags'] = tags
    result = None
    if stream_to_update:
        if deleted_items:
            result = self.update_image_stream_status(stream)
        if self.params.get('prune_registry'):
            self.delete_manifests(name, manifests_to_delete)
    return (result, images_to_delete)