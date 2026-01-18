from __future__ import (absolute_import, division, print_function)
from ansible.playbook import Play
from ansible.playbook.block import Block
from ansible.playbook.role import Role
from ansible.playbook.task import Task
from ansible.utils.display import Display
def get_reserved_names(include_private=True):
    """ this function returns the list of reserved names associated with play objects"""
    public = set()
    private = set()
    result = set()
    class_list = [Play, Role, Block, Task]
    for aclass in class_list:
        for name, attr in aclass.fattributes.items():
            if attr.private:
                private.add(name)
            else:
                public.add(name)
    if 'action' in public:
        public.add('local_action')
    if 'loop' in private or 'loop' in public:
        public.add('with_')
    if include_private:
        result = public.union(private)
    else:
        result = public
    return result