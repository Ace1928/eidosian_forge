from __future__ import (absolute_import, division, print_function)
from os.path import basename
import ansible.constants as C
from ansible.errors import AnsibleParserError
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.task_include import TaskInclude
from ansible.playbook.role import Role
from ansible.playbook.role.include import RoleInclude
from ansible.utils.display import Display
from ansible.module_utils.six import string_types
from ansible.template import Templar
def get_block_list(self, play=None, variable_manager=None, loader=None):
    if play is None:
        myplay = self._parent._play
    else:
        myplay = play
    ri = RoleInclude.load(self._role_name, play=myplay, variable_manager=variable_manager, loader=loader, collection_list=self.collections)
    ri.vars |= self.vars
    if variable_manager is not None:
        available_variables = variable_manager.get_vars(play=myplay, task=self)
    else:
        available_variables = {}
    templar = Templar(loader=loader, variables=available_variables)
    from_files = templar.template(self._from_files)
    actual_role = Role.load(ri, myplay, parent_role=self._parent_role, from_files=from_files, from_include=True, validate=self.rolespec_validate, public=self.public, static=self.statically_loaded)
    actual_role._metadata.allow_duplicates = self.allow_duplicates
    myplay.roles.append(actual_role)
    self._role_path = actual_role._role_path
    dep_chain = actual_role.get_dep_chain()
    p_block = self.build_parent_block()
    p_block.collections = actual_role.collections
    blocks = actual_role.compile(play=myplay, dep_chain=dep_chain)
    for b in blocks:
        b._parent = p_block
        b.collections = actual_role.collections
    handlers = actual_role.get_handler_blocks(play=myplay, dep_chain=dep_chain)
    for h in handlers:
        h._parent = p_block
    myplay.handlers = myplay.handlers + handlers
    return (blocks, handlers)