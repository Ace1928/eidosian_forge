from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
@staticmethod
def perform_migrations(templateinstances):
    ti_list = []
    ti_to_be_migrated = []
    ti_list = templateinstances.get('kind') == 'TemplateInstanceList' and templateinstances.get('items') or [templateinstances]
    for ti_elem in ti_list:
        objects = ti_elem['status'].get('objects')
        if objects:
            for i, obj in enumerate(objects):
                object_type = obj['ref']['kind']
                if object_type in transforms.keys() and obj['ref'].get('apiVersion') != transforms[object_type]:
                    ti_elem['status']['objects'][i]['ref']['apiVersion'] = transforms[object_type]
                    ti_to_be_migrated.append(ti_elem)
    return ti_to_be_migrated