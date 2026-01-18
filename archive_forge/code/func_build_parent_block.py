from __future__ import (absolute_import, division, print_function)
import ansible.constants as C
from ansible.errors import AnsibleParserError
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
def build_parent_block(self):
    """
        This method is used to create the parent block for the included tasks
        when ``apply`` is specified
        """
    apply_attrs = self.args.pop('apply', {})
    if apply_attrs:
        apply_attrs['block'] = []
        p_block = Block.load(apply_attrs, play=self._parent._play, task_include=self, role=self._role, variable_manager=self._variable_manager, loader=self._loader)
    else:
        p_block = self
    return p_block