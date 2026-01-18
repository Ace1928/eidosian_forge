from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def generate_substructure_spec(self, scope, code):
    if self.is_empty(scope):
        return
    from .Code import UtilityCode
    code.globalstate.use_utility_code(UtilityCode.load_cached('IncludeStructmemberH', 'ModuleSetupCode.c'))
    code.putln('static struct PyMemberDef %s[] = {' % self.substructure_cname(scope))
    for member_entry in self.get_member_specs(scope):
        if member_entry:
            code.putln(member_entry)
    code.putln('{NULL, 0, 0, 0, NULL}')
    code.putln('};')