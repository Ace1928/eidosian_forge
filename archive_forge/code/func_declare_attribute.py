from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
def declare_attribute(self, attribute, env, pos):
    from . import MemoryView, Options
    scope = self.scope
    if attribute == 'shape':
        scope.declare_var('shape', c_array_type(c_py_ssize_t_type, Options.buffer_max_dims), pos, cname='shape', is_cdef=1)
    elif attribute == 'strides':
        scope.declare_var('strides', c_array_type(c_py_ssize_t_type, Options.buffer_max_dims), pos, cname='strides', is_cdef=1)
    elif attribute == 'suboffsets':
        scope.declare_var('suboffsets', c_array_type(c_py_ssize_t_type, Options.buffer_max_dims), pos, cname='suboffsets', is_cdef=1)
    elif attribute in ('copy', 'copy_fortran'):
        ndim = len(self.axes)
        follow_dim = [('direct', 'follow')]
        contig_dim = [('direct', 'contig')]
        to_axes_c = follow_dim * (ndim - 1) + contig_dim
        to_axes_f = contig_dim + follow_dim * (ndim - 1)
        dtype = self.dtype
        if dtype.is_cv_qualified:
            dtype = dtype.cv_base_type
        to_memview_c = MemoryViewSliceType(dtype, to_axes_c)
        to_memview_f = MemoryViewSliceType(dtype, to_axes_f)
        for to_memview, cython_name in [(to_memview_c, 'copy'), (to_memview_f, 'copy_fortran')]:
            copy_func_type = CFuncType(to_memview, [CFuncTypeArg('memviewslice', self, None)])
            copy_cname = MemoryView.copy_c_or_fortran_cname(to_memview)
            entry = scope.declare_cfunction(cython_name, copy_func_type, pos=pos, defining=1, cname=copy_cname)
            utility = MemoryView.get_copy_new_utility(pos, self, to_memview)
            env.use_utility_code(utility)
        MemoryView.use_cython_array_utility_code(env)
    elif attribute in ('is_c_contig', 'is_f_contig'):
        for c_or_f, cython_name in (('C', 'is_c_contig'), ('F', 'is_f_contig')):
            is_contig_name = MemoryView.get_is_contig_func_name(c_or_f, self.ndim)
            cfunctype = CFuncType(return_type=c_bint_type, args=[CFuncTypeArg('memviewslice', self, None)], exception_value='-1')
            entry = scope.declare_cfunction(cython_name, cfunctype, pos=pos, defining=1, cname=is_contig_name)
            entry.utility_code_definition = MemoryView.get_is_contig_utility(c_or_f, self.ndim)
    return True