from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.utils.display import Display
def set_attributes_from_plugin(self, plugin):
    options = C.config.get_configuration_definitions(plugin.plugin_type, plugin._load_name)
    for option in options:
        if option:
            flag = options[option].get('name')
            if flag:
                setattr(self, flag, plugin.get_option(flag))