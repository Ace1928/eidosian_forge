from __future__ import (absolute_import, division, print_function)
import re
import operator
from functools import reduce
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible.module_utils._text import to_native
def resolve_imagestream_triggers(self, existing, definition):
    existing_triggers = existing.get('spec', {}).get('triggers')
    new_triggers = definition['spec']['triggers']
    existing_containers = existing.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
    new_containers = definition.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
    for i, trigger in enumerate(new_triggers):
        if trigger.get('type') == 'ImageChange' and trigger.get('imageChangeParams'):
            names = trigger['imageChangeParams'].get('containerNames', [])
            for name in names:
                old_container_index = self.get_index({'name': name}, existing_containers, ['name'])
                new_container_index = self.get_index({'name': name}, new_containers, ['name'])
                if old_container_index is not None and new_container_index is not None:
                    image = existing['spec']['template']['spec']['containers'][old_container_index]['image']
                    definition['spec']['template']['spec']['containers'][new_container_index]['image'] = image
                existing_index = self.get_index(trigger['imageChangeParams'], [x.get('imageChangeParams') for x in existing_triggers], ['containerNames'])
                if existing_index is not None:
                    existing_image = existing_triggers[existing_index].get('imageChangeParams', {}).get('lastTriggeredImage')
                    if existing_image:
                        definition['spec']['triggers'][i]['imageChangeParams']['lastTriggeredImage'] = existing_image
                    existing_from = existing_triggers[existing_index].get('imageChangeParams', {}).get('from', {})
                    new_from = trigger['imageChangeParams'].get('from', {})
                    existing_namespace = existing_from.get('namespace')
                    existing_name = existing_from.get('name', False)
                    new_name = new_from.get('name', True)
                    add_namespace = existing_namespace and 'namespace' not in new_from.keys() and (existing_name == new_name)
                    if add_namespace:
                        definition['spec']['triggers'][i]['imageChangeParams']['from']['namespace'] = existing_from['namespace']
    return definition