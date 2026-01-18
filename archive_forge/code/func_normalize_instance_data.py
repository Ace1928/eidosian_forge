from __future__ import absolute_import, division, print_function
import yaml
from ansible.module_utils.basic import missing_required_lib
from ansible.plugins.inventory import (AnsibleError, BaseInventoryPlugin,
from jinja2 import Template
from ..module_utils.cloudstack import HAS_LIB_CS
def normalize_instance_data(self, instance):
    inventory_instance_str = self._normalization_template.render(instance=instance)
    inventory_instance = yaml.load(inventory_instance_str, Loader=yaml.FullLoader)
    return inventory_instance['instance']