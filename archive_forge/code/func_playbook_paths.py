from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import add_metaclass
@playbook_paths.setter
def playbook_paths(cls, value):
    cls._require_finder()
    cls._collection_finder.set_playbook_paths(value)