from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def generate_buffer_slice_code(self, code, indices, dst, dst_type, have_gil, have_slices, directives):
    """
        Slice a memoryviewslice.

        indices     - list of index nodes. If not a SliceNode, or NoneNode,
                      then it must be coercible to Py_ssize_t

        Simply call __pyx_memoryview_slice_memviewslice with the right
        arguments, unless the dimension is omitted or a bare ':', in which
        case we copy over the shape/strides/suboffsets attributes directly
        for that dimension.
        """
    src = self.cname
    code.putln('%(dst)s.data = %(src)s.data;' % locals())
    code.putln('%(dst)s.memview = %(src)s.memview;' % locals())
    code.put_incref_memoryviewslice(dst, dst_type, have_gil=have_gil)
    all_dimensions_direct = all((access == 'direct' for access, packing in self.type.axes))
    suboffset_dim_temp = []

    def get_suboffset_dim():
        if not suboffset_dim_temp:
            suboffset_dim = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
            code.putln('%s = -1;' % suboffset_dim)
            suboffset_dim_temp.append(suboffset_dim)
        return suboffset_dim_temp[0]
    dim = -1
    new_ndim = 0
    for index in indices:
        if index.is_none:
            for attrib, value in [('shape', 1), ('strides', 0), ('suboffsets', -1)]:
                code.putln('%s.%s[%d] = %d;' % (dst, attrib, new_ndim, value))
            new_ndim += 1
            continue
        dim += 1
        access, packing = self.type.axes[dim]
        if index.is_slice:
            d = dict(locals())
            for s in 'start stop step'.split():
                idx = getattr(index, s)
                have_idx = d['have_' + s] = not idx.is_none
                d[s] = idx.result() if have_idx else '0'
            if not (d['have_start'] or d['have_stop'] or d['have_step']):
                d['access'] = access
                util_name = 'SimpleSlice'
            else:
                util_name = 'ToughSlice'
                d['error_goto'] = code.error_goto(index.pos)
            new_ndim += 1
        else:
            idx = index.result()
            indirect = access != 'direct'
            if indirect:
                generic = access == 'full'
                if new_ndim != 0:
                    return error(index.pos, 'All preceding dimensions must be indexed and not sliced')
            d = dict(locals(), wraparound=int(directives['wraparound']), boundscheck=int(directives['boundscheck']))
            if d['boundscheck']:
                d['error_goto'] = code.error_goto(index.pos)
            util_name = 'SliceIndex'
        _, impl = TempitaUtilityCode.load_as_string(util_name, 'MemoryView_C.c', context=d)
        code.put(impl)
    if suboffset_dim_temp:
        code.funcstate.release_temp(suboffset_dim_temp[0])