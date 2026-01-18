from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab
def put_buffer_lookup_code(entry, index_signeds, index_cnames, directives, pos, code, negative_indices, in_nogil_context):
    """
    Generates code to process indices and calculate an offset into
    a buffer. Returns a C string which gives a pointer which can be
    read from or written to at will (it is an expression so caller should
    store it in a temporary if it is used more than once).

    As the bounds checking can have any number of combinations of unsigned
    arguments, smart optimizations etc. we insert it directly in the function
    body. The lookup however is delegated to a inline function that is instantiated
    once per ndim (lookup with suboffsets tend to get quite complicated).

    entry is a BufferEntry
    """
    negative_indices = directives['wraparound'] and negative_indices
    if directives['boundscheck']:
        failed_dim_temp = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
        code.putln('%s = -1;' % failed_dim_temp)
        for dim, (signed, cname, shape) in enumerate(zip(index_signeds, index_cnames, entry.get_buf_shapevars())):
            if signed != 0:
                code.putln('if (%s < 0) {' % cname)
                if negative_indices:
                    code.putln('%s += %s;' % (cname, shape))
                    code.putln('if (%s) %s = %d;' % (code.unlikely('%s < 0' % cname), failed_dim_temp, dim))
                else:
                    code.putln('%s = %d;' % (failed_dim_temp, dim))
                code.put('} else ')
            if signed != 0:
                cast = ''
            else:
                cast = '(size_t)'
            code.putln('if (%s) %s = %d;' % (code.unlikely('%s >= %s%s' % (cname, cast, shape)), failed_dim_temp, dim))
        if in_nogil_context:
            code.globalstate.use_utility_code(raise_indexerror_nogil)
            func = '__Pyx_RaiseBufferIndexErrorNogil'
        else:
            code.globalstate.use_utility_code(raise_indexerror_code)
            func = '__Pyx_RaiseBufferIndexError'
        code.putln('if (%s) {' % code.unlikely('%s != -1' % failed_dim_temp))
        code.putln('%s(%s);' % (func, failed_dim_temp))
        code.putln(code.error_goto(pos))
        code.putln('}')
        code.funcstate.release_temp(failed_dim_temp)
    elif negative_indices:
        for signed, cname, shape in zip(index_signeds, index_cnames, entry.get_buf_shapevars()):
            if signed != 0:
                code.putln('if (%s < 0) %s += %s;' % (cname, cname, shape))
    return entry.generate_buffer_lookup_code(code, index_cnames)