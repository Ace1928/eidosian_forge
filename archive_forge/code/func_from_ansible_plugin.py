from __future__ import absolute_import, division, print_function
import os
@classmethod
def from_ansible_plugin(cls, plugin):
    return cls(getter=plugin.get_option, setter=plugin.set_option, haver=plugin.has_option if hasattr(plugin, 'has_option') else None)