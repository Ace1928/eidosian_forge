import sys
from . import model
from .error import FFIError
def resolve_common_type(parser, commontype):
    try:
        return _CACHE[commontype]
    except KeyError:
        cdecl = COMMON_TYPES.get(commontype, commontype)
        if not isinstance(cdecl, str):
            result, quals = (cdecl, 0)
        elif cdecl in model.PrimitiveType.ALL_PRIMITIVE_TYPES:
            result, quals = (model.PrimitiveType(cdecl), 0)
        elif cdecl == 'set-unicode-needed':
            raise FFIError('The Windows type %r is only available after you call ffi.set_unicode()' % (commontype,))
        else:
            if commontype == cdecl:
                raise FFIError('Unsupported type: %r.  Please look at http://cffi.readthedocs.io/en/latest/cdef.html#ffi-cdef-limitations and file an issue if you think this type should really be supported.' % (commontype,))
            result, quals = parser.parse_type_and_quals(cdecl)
        assert isinstance(result, model.BaseTypeByIdentity)
        _CACHE[commontype] = (result, quals)
        return (result, quals)