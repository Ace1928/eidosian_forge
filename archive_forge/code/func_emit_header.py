import logging
import os
import sys
from llvmlite import ir
from llvmlite.binding import Linkage
from numba.pycc import llvm_types as lt
from numba.core.cgutils import create_constant_array
from numba.core.compiler import compile_extra, Flags
from numba.core.compiler_lock import global_compiler_lock
from numba.core.registry import cpu_target
from numba.core.runtime import nrtdynmod
from numba.core import cgutils
def emit_header(self, output):
    fname, ext = os.path.splitext(output)
    with open(fname + '.h', 'w') as fout:
        fout.write(get_header())
        fout.write('\n/* Prototypes */\n')
        for export_entry in self.export_entries:
            name = export_entry.symbol
            restype = self.emit_type(export_entry.signature.return_type)
            args = ', '.join((self.emit_type(argtype) for argtype in export_entry.signature.args))
            fout.write('extern %s %s(%s);\n' % (restype, name, args))