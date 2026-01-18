from collections import ChainMap
from jinja2.utils import missing
from ansible.errors import AnsibleError, AnsibleUndefinedVariable
from ansible.module_utils.common.text.converters import to_native
def add_locals(self, locals):
    """If locals are provided, create a copy of self containing those
        locals in addition to what is already in this variable proxy.
        """
    if locals is None:
        return self
    current_locals = self.maps[0]
    current_globals = self.maps[2]
    new_locals = current_locals | locals
    return AnsibleJ2Vars(self._templar, current_globals, locals=new_locals)