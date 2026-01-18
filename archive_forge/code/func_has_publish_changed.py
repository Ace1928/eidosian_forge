from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def has_publish_changed(self, old_publish):
    if self.publish is None:
        return False
    old_publish = old_publish or []
    if len(self.publish) != len(old_publish):
        return True

    def publish_sorter(item):
        return (item.get('published_port') or 0, item.get('target_port') or 0, item.get('protocol') or '')
    publish = sorted(self.publish, key=publish_sorter)
    old_publish = sorted(old_publish, key=publish_sorter)
    for publish_item, old_publish_item in zip(publish, old_publish):
        ignored_keys = set()
        if not publish_item.get('mode'):
            ignored_keys.add('mode')
        filtered_old_publish_item = dict(((k, v) for k, v in old_publish_item.items() if k not in ignored_keys))
        filtered_publish_item = dict(((k, v) for k, v in publish_item.items() if k not in ignored_keys))
        if filtered_publish_item != filtered_old_publish_item:
            return True
    return False