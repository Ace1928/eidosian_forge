from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
def generate_type_import_call(self, type, code, import_generator, error_code=None, error_pos=None):
    if type.typedef_flag:
        objstruct = type.objstruct_cname
    else:
        objstruct = 'struct %s' % type.objstruct_cname
    sizeof_objstruct = objstruct
    module_name = type.module_name
    condition = replacement = None
    if module_name not in ('__builtin__', 'builtins'):
        module_name = '"%s"' % module_name
    elif type.name in Code.ctypedef_builtins_map:
        ctypename = Code.ctypedef_builtins_map[type.name]
        code.putln('%s = %s;' % (type.typeptr_cname, ctypename))
        return
    else:
        module_name = '__Pyx_BUILTIN_MODULE_NAME'
        if type.name in Code.non_portable_builtins_map:
            condition, replacement = Code.non_portable_builtins_map[type.name]
        if objstruct in Code.basicsize_builtins_map:
            sizeof_objstruct = Code.basicsize_builtins_map[objstruct]
    if not error_code:
        assert error_pos is not None
        error_code = code.error_goto(error_pos)
    module = import_generator.imported_module(module_name, error_code)
    code.put('%s = __Pyx_ImportType_%s(%s, %s,' % (type.typeptr_cname, Naming.cyversion, module, module_name))
    type_name = type.name.as_c_string_literal()
    if condition and replacement:
        code.putln('')
        code.putln('#if %s' % condition)
        code.putln('"%s",' % replacement)
        code.putln('#else')
        code.putln('%s,' % type_name)
        code.putln('#endif')
    else:
        code.put(' %s, ' % type_name)
    if sizeof_objstruct != objstruct:
        if not condition:
            code.putln('')
        code.putln('#if defined(PYPY_VERSION_NUM) && PYPY_VERSION_NUM < 0x050B0000')
        code.putln('sizeof(%s), __PYX_GET_STRUCT_ALIGNMENT_%s(%s),' % (objstruct, Naming.cyversion, objstruct))
        code.putln('#elif CYTHON_COMPILING_IN_LIMITED_API')
        code.putln('sizeof(%s), __PYX_GET_STRUCT_ALIGNMENT_%s(%s),' % (objstruct, Naming.cyversion, objstruct))
        code.putln('#else')
        code.putln('sizeof(%s), __PYX_GET_STRUCT_ALIGNMENT_%s(%s),' % (sizeof_objstruct, Naming.cyversion, sizeof_objstruct))
        code.putln('#endif')
    else:
        code.put('sizeof(%s), __PYX_GET_STRUCT_ALIGNMENT_%s(%s),' % (objstruct, Naming.cyversion, objstruct))
    if type.check_size and type.check_size in ('error', 'warn', 'ignore'):
        check_size = type.check_size
    elif not type.is_external or type.is_subclassed:
        check_size = 'error'
    else:
        raise RuntimeError("invalid value for check_size '%s' when compiling %s.%s" % (type.check_size, module_name, type.name))
    code.put('__Pyx_ImportType_CheckSize_%s_%s);' % (check_size.title(), Naming.cyversion))
    code.putln(' if (!%s) %s' % (type.typeptr_cname, error_code))