from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def declare_c_class(self, name, pos, defining=0, implementing=0, module_name=None, base_type=None, objstruct_cname=None, typeobj_cname=None, typeptr_cname=None, visibility='private', typedef_flag=0, api=0, check_size=None, buffer_defaults=None, shadow=0):
    if typedef_flag and visibility != 'extern':
        if not (visibility == 'public' or api):
            warning(pos, "ctypedef only valid for 'extern' , 'public', and 'api'", 2)
        objtypedef_cname = objstruct_cname
        typedef_flag = 0
    else:
        objtypedef_cname = None
    entry = self.lookup_here(name)
    if entry and (not shadow):
        type = entry.type
        if not (entry.is_type and type.is_extension_type):
            entry = None
        else:
            scope = type.scope
            if typedef_flag and (not scope or scope.defined):
                self.check_previous_typedef_flag(entry, typedef_flag, pos)
            if scope and scope.defined or (base_type and type.base_type):
                if base_type and base_type is not type.base_type:
                    error(pos, 'Base type does not match previous declaration')
            if base_type and (not type.base_type):
                type.base_type = base_type
    if not entry or shadow:
        type = PyrexTypes.PyExtensionType(name, typedef_flag, base_type, visibility == 'extern', check_size=check_size)
        type.pos = pos
        type.buffer_defaults = buffer_defaults
        if objtypedef_cname is not None:
            type.objtypedef_cname = objtypedef_cname
        if visibility == 'extern':
            type.module_name = module_name
        else:
            type.module_name = self.qualified_name
        if typeptr_cname:
            type.typeptr_cname = typeptr_cname
        else:
            type.typeptr_cname = self.mangle(Naming.typeptr_prefix, name)
        entry = self.declare_type(name, type, pos, visibility=visibility, defining=0, shadow=shadow)
        entry.is_cclass = True
        if objstruct_cname:
            type.objstruct_cname = objstruct_cname
        elif not entry.in_cinclude:
            type.objstruct_cname = self.mangle(Naming.objstruct_prefix, name)
        else:
            error(entry.pos, "Object name required for 'public' or 'extern' C class")
        self.attach_var_entry_to_c_class(entry)
        self.c_class_entries.append(entry)
    if not type.scope:
        if defining or implementing:
            scope = CClassScope(name=name, outer_scope=self, visibility=visibility, parent_type=type)
            scope.directives = self.directives.copy()
            if base_type and base_type.scope:
                scope.declare_inherited_c_attributes(base_type.scope)
            type.set_scope(scope)
            self.type_entries.append(entry)
    elif defining and type.scope.defined:
        error(pos, "C class '%s' already defined" % name)
    elif implementing and type.scope.implemented:
        error(pos, "C class '%s' already implemented" % name)
    if defining:
        entry.defined_in_pxd = 1
    if implementing:
        entry.pos = pos
    if visibility != 'private' and entry.visibility != visibility:
        error(pos, "Class '%s' previously declared as '%s'" % (name, entry.visibility))
    if api:
        entry.api = 1
    if objstruct_cname:
        if type.objstruct_cname and type.objstruct_cname != objstruct_cname:
            error(pos, 'Object struct name differs from previous declaration')
        type.objstruct_cname = objstruct_cname
    if typeobj_cname:
        if type.typeobj_cname and type.typeobj_cname != typeobj_cname:
            error(pos, 'Type object name differs from previous declaration')
        type.typeobj_cname = typeobj_cname
    if self.directives.get('final'):
        entry.type.is_final_type = True
    collection_type = self.directives.get('collection_type')
    if collection_type:
        from .UtilityCode import NonManglingModuleScope
        if not isinstance(self, NonManglingModuleScope):
            error(pos, "'collection_type' is not a public cython directive")
    if collection_type == 'sequence':
        entry.type.has_sequence_flag = True
    entry.used = True
    return entry