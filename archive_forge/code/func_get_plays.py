from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.playbook.play import Play
from ansible.playbook.playbook_include import PlaybookInclude
from ansible.plugins.loader import add_all_plugin_dirs
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath
def get_plays(self):
    return self._entries[:]