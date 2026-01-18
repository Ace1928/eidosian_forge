from __future__ import absolute_import
import copy
from . import (ExprNodes, PyrexTypes, MemoryView,
from .ExprNodes import CloneNode, ProxyNode, TupleNode
from .Nodes import FuncDefNode, CFuncDefNode, StatListNode, DefNode
from ..Utils import OrderedSet
from .Errors import error, CannotSpecialize
def copy_cdef(self, env):
    """
        Create a copy of the original c(p)def function for all specialized
        versions.
        """
    permutations = self.node.type.get_all_specialized_permutations()
    self.orig_py_func = orig_py_func = self.node.py_func
    self.node.py_func = None
    if orig_py_func:
        env.pyfunc_entries.remove(orig_py_func.entry)
    fused_types = self.node.type.get_fused_types()
    self.fused_compound_types = fused_types
    new_cfunc_entries = []
    for cname, fused_to_specific in permutations:
        copied_node = copy.deepcopy(self.node)
        try:
            type = copied_node.type.specialize(fused_to_specific)
        except CannotSpecialize:
            error(copied_node.pos, 'Return type is a fused type that cannot be determined from the function arguments')
            self.py_func = None
            return
        entry = copied_node.entry
        type.specialize_entry(entry, cname)
        for i, orig_entry in enumerate(env.cfunc_entries):
            if entry.cname == orig_entry.cname and type.same_as_resolved_type(orig_entry.type):
                copied_node.entry = env.cfunc_entries[i]
                if not copied_node.entry.func_cname:
                    copied_node.entry.func_cname = entry.func_cname
                entry = copied_node.entry
                type = entry.type
                break
        else:
            new_cfunc_entries.append(entry)
        copied_node.type = type
        entry.type, type.entry = (type, entry)
        entry.used = entry.used or self.node.entry.defined_in_pxd or env.is_c_class_scope or entry.is_cmethod
        if self.node.cfunc_declarator.optional_arg_count:
            self.node.cfunc_declarator.declare_optional_arg_struct(type, env, fused_cname=cname)
        copied_node.return_type = type.return_type
        self.create_new_local_scope(copied_node, env, fused_to_specific)
        self._specialize_function_args(copied_node.cfunc_declarator.args, fused_to_specific)
        copied_node.declare_cpdef_wrapper(env)
        if copied_node.py_func:
            env.pyfunc_entries.remove(copied_node.py_func.entry)
            self.specialize_copied_def(copied_node.py_func, cname, self.node.entry.as_variable, fused_to_specific, fused_types)
        if not self.replace_fused_typechecks(copied_node):
            break
    if self.node.entry in env.cfunc_entries:
        cindex = env.cfunc_entries.index(self.node.entry)
        env.cfunc_entries[cindex:cindex + 1] = new_cfunc_entries
    else:
        env.cfunc_entries.extend(new_cfunc_entries)
    if orig_py_func:
        self.py_func = self.make_fused_cpdef(orig_py_func, env, is_def=False)
    else:
        self.py_func = orig_py_func