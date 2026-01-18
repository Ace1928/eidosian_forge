from __future__ import (absolute_import, division, print_function)
import ansible.constants as C
from ansible.errors import AnsibleParserError
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.conditional import Conditional
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.delegatable import Delegatable
from ansible.playbook.helpers import load_list_of_tasks
from ansible.playbook.notifiable import Notifiable
from ansible.playbook.role import Role
from ansible.playbook.taggable import Taggable
from ansible.utils.sentinel import Sentinel
def filter_tagged_tasks(self, all_vars):
    """
        Creates a new block, with task lists filtered based on the tags.
        """

    def evaluate_and_append_task(target):
        tmp_list = []
        for task in target:
            if isinstance(task, Block):
                filtered_block = evaluate_block(task)
                if filtered_block.has_tasks():
                    tmp_list.append(filtered_block)
            elif task.action in C._ACTION_META and task.implicit or task.evaluate_tags(self._play.only_tags, self._play.skip_tags, all_vars=all_vars):
                tmp_list.append(task)
        return tmp_list

    def evaluate_block(block):
        new_block = block.copy(exclude_parent=True, exclude_tasks=True)
        new_block._parent = block._parent
        new_block.block = evaluate_and_append_task(block.block)
        new_block.rescue = evaluate_and_append_task(block.rescue)
        new_block.always = evaluate_and_append_task(block.always)
        return new_block
    return evaluate_block(self)