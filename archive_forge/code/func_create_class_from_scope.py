from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def create_class_from_scope(self, node, target_module_scope, inner_node=None):
    if node.is_generator:
        for scope in node.local_scope.iter_local_scopes():
            for entry in scope.entries.values():
                if not (entry.from_closure or entry.is_pyglobal or entry.is_cglobal):
                    entry.in_closure = True
    from_closure, in_closure = self.find_entries_used_in_closures(node)
    in_closure.sort()
    node.needs_closure = False
    node.needs_outer_scope = False
    func_scope = node.local_scope
    cscope = node.entry.scope
    while cscope.is_py_class_scope or cscope.is_c_class_scope:
        cscope = cscope.outer_scope
    if not from_closure and (self.path or inner_node):
        if not inner_node:
            if not node.py_cfunc_node:
                raise InternalError('DefNode does not have assignment node')
            inner_node = node.py_cfunc_node
        inner_node.needs_closure_code = False
        node.needs_outer_scope = False
    if node.is_generator:
        pass
    elif not in_closure and (not from_closure):
        return
    elif not in_closure:
        func_scope.is_passthrough = True
        func_scope.scope_class = cscope.scope_class
        node.needs_outer_scope = True
        return
    as_name = '%s_%s' % (target_module_scope.next_id(Naming.closure_class_prefix), node.entry.cname.replace('.', '__'))
    as_name = EncodedString(as_name)
    entry = target_module_scope.declare_c_class(name=as_name, pos=node.pos, defining=True, implementing=True)
    entry.type.is_final_type = True
    func_scope.scope_class = entry
    class_scope = entry.type.scope
    class_scope.is_internal = True
    class_scope.is_closure_class_scope = True
    if node.is_async_def or node.is_generator:
        class_scope.directives['no_gc_clear'] = True
    if Options.closure_freelist_size:
        class_scope.directives['freelist'] = Options.closure_freelist_size
    if from_closure:
        assert cscope.is_closure_scope
        class_scope.declare_var(pos=node.pos, name=Naming.outer_scope_cname, cname=Naming.outer_scope_cname, type=cscope.scope_class.type, is_cdef=True)
        node.needs_outer_scope = True
    for name, entry in in_closure:
        closure_entry = class_scope.declare_var(pos=entry.pos, name=entry.name if not entry.in_subscope else None, cname=entry.cname, type=entry.type, is_cdef=True)
        if entry.is_declared_generic:
            closure_entry.is_declared_generic = 1
    node.needs_closure = True
    target_module_scope.check_c_class(func_scope.scope_class)