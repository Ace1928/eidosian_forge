from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def analyse_default_args(self, env):
    """
        Handle non-literal function's default arguments.
        """
    nonliteral_objects = []
    nonliteral_other = []
    default_args = []
    default_kwargs = []
    annotations = []
    must_use_constants = env.is_c_class_scope or (self.def_node.is_wrapper and env.is_module_scope)
    for arg in self.def_node.args:
        if arg.default:
            if not must_use_constants:
                if arg.default.is_literal:
                    arg.default = DefaultLiteralArgNode(arg.pos, arg.default)
                    if arg.default.type:
                        arg.default = arg.default.coerce_to(arg.type, env)
                else:
                    arg.is_dynamic = True
                    if arg.type.is_pyobject:
                        nonliteral_objects.append(arg)
                    else:
                        nonliteral_other.append(arg)
            if arg.default.type and arg.default.type.can_coerce_to_pyobject(env):
                if arg.kw_only:
                    default_kwargs.append(arg)
                else:
                    default_args.append(arg)
        if arg.annotation:
            arg.annotation = arg.annotation.analyse_types(env)
            annotations.append((arg.pos, arg.name, arg.annotation.string))
    for arg in (self.def_node.star_arg, self.def_node.starstar_arg):
        if arg and arg.annotation:
            arg.annotation = arg.annotation.analyse_types(env)
            annotations.append((arg.pos, arg.name, arg.annotation.string))
    annotation = self.def_node.return_type_annotation
    if annotation:
        self.def_node.return_type_annotation = annotation.analyse_types(env)
        annotations.append((annotation.pos, StringEncoding.EncodedString('return'), annotation.string))
    if nonliteral_objects or nonliteral_other:
        module_scope = env.global_scope()
        cname = module_scope.next_id(Naming.defaults_struct_prefix)
        scope = Symtab.StructOrUnionScope(cname)
        self.defaults = []
        for arg in nonliteral_objects:
            type_ = arg.type
            if type_.is_buffer:
                type_ = type_.base
            entry = scope.declare_var(arg.name, type_, None, Naming.arg_prefix + arg.name, allow_pyobject=True)
            self.defaults.append((arg, entry))
        for arg in nonliteral_other:
            entry = scope.declare_var(arg.name, arg.type, None, Naming.arg_prefix + arg.name, allow_pyobject=False, allow_memoryview=True)
            self.defaults.append((arg, entry))
        entry = module_scope.declare_struct_or_union(None, 'struct', scope, 1, None, cname=cname)
        self.defaults_struct = scope
        self.defaults_pyobjects = len(nonliteral_objects)
        for arg, entry in self.defaults:
            arg.default_value = '%s->%s' % (Naming.dynamic_args_cname, entry.cname)
        self.def_node.defaults_struct = self.defaults_struct.name
    if default_args or default_kwargs:
        if self.defaults_struct is None:
            if default_args:
                defaults_tuple = TupleNode(self.pos, args=[arg.default for arg in default_args])
                self.defaults_tuple = defaults_tuple.analyse_types(env).coerce_to_pyobject(env)
            if default_kwargs:
                defaults_kwdict = DictNode(self.pos, key_value_pairs=[DictItemNode(arg.pos, key=IdentifierStringNode(arg.pos, value=arg.name), value=arg.default) for arg in default_kwargs])
                self.defaults_kwdict = defaults_kwdict.analyse_types(env)
        elif not self.specialized_cpdefs:
            if default_args:
                defaults_tuple = DefaultsTupleNode(self.pos, default_args, self.defaults_struct)
            else:
                defaults_tuple = NoneNode(self.pos)
            if default_kwargs:
                defaults_kwdict = DefaultsKwDictNode(self.pos, default_kwargs, self.defaults_struct)
            else:
                defaults_kwdict = NoneNode(self.pos)
            defaults_getter = Nodes.DefNode(self.pos, args=[], star_arg=None, starstar_arg=None, body=Nodes.ReturnStatNode(self.pos, return_type=py_object_type, value=TupleNode(self.pos, args=[defaults_tuple, defaults_kwdict])), decorators=None, name=StringEncoding.EncodedString('__defaults__'))
            module_scope = env.global_scope()
            defaults_getter.analyse_declarations(module_scope)
            defaults_getter = defaults_getter.analyse_expressions(module_scope)
            defaults_getter.body = defaults_getter.body.analyse_expressions(defaults_getter.local_scope)
            defaults_getter.py_wrapper_required = False
            defaults_getter.pymethdef_required = False
            self.def_node.defaults_getter = defaults_getter
    if annotations:
        annotations_dict = DictNode(self.pos, key_value_pairs=[DictItemNode(pos, key=IdentifierStringNode(pos, value=name), value=value) for pos, name, value in annotations])
        self.annotations_dict = annotations_dict.analyse_types(env)