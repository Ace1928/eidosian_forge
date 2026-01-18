import ctypes, ctypes.util, operator, sys
from . import model
def new_pointer_type(self, BItem):
    getbtype = self.ffi._get_cached_btype
    if BItem is getbtype(model.PrimitiveType('char')):
        kind = 'charp'
    elif BItem in (getbtype(model.PrimitiveType('signed char')), getbtype(model.PrimitiveType('unsigned char'))):
        kind = 'bytep'
    elif BItem is getbtype(model.void_type):
        kind = 'voidp'
    else:
        kind = 'generic'

    class CTypesPtr(CTypesGenericPtr):
        __slots__ = ['_own']
        if kind == 'charp':
            __slots__ += ['__as_strbuf']
        _BItem = BItem
        if hasattr(BItem, '_ctype'):
            _ctype = ctypes.POINTER(BItem._ctype)
            _bitem_size = ctypes.sizeof(BItem._ctype)
        else:
            _ctype = ctypes.c_void_p
        if issubclass(BItem, CTypesGenericArray):
            _reftypename = BItem._get_c_name('(* &)')
        else:
            _reftypename = BItem._get_c_name(' * &')

        def __init__(self, init):
            ctypeobj = BItem._create_ctype_obj(init)
            if kind == 'charp':
                self.__as_strbuf = ctypes.create_string_buffer(ctypeobj.value + b'\x00')
                self._as_ctype_ptr = ctypes.cast(self.__as_strbuf, self._ctype)
            else:
                self._as_ctype_ptr = ctypes.pointer(ctypeobj)
            self._address = ctypes.cast(self._as_ctype_ptr, ctypes.c_void_p).value
            self._own = True

        def __add__(self, other):
            if isinstance(other, (int, long)):
                return self._new_pointer_at(self._address + other * self._bitem_size)
            else:
                return NotImplemented

        def __sub__(self, other):
            if isinstance(other, (int, long)):
                return self._new_pointer_at(self._address - other * self._bitem_size)
            elif type(self) is type(other):
                return (self._address - other._address) // self._bitem_size
            else:
                return NotImplemented

        def __getitem__(self, index):
            if getattr(self, '_own', False) and index != 0:
                raise IndexError
            return BItem._from_ctypes(self._as_ctype_ptr[index])

        def __setitem__(self, index, value):
            self._as_ctype_ptr[index] = BItem._to_ctypes(value)
        if kind == 'charp' or kind == 'voidp':

            @classmethod
            def _arg_to_ctypes(cls, *value):
                if value and isinstance(value[0], bytes):
                    return ctypes.c_char_p(value[0])
                else:
                    return super(CTypesPtr, cls)._arg_to_ctypes(*value)
        if kind == 'charp' or kind == 'bytep':

            def _to_string(self, maxlen):
                if maxlen < 0:
                    maxlen = sys.maxsize
                p = ctypes.cast(self._as_ctype_ptr, ctypes.POINTER(ctypes.c_char))
                n = 0
                while n < maxlen and p[n] != b'\x00':
                    n += 1
                return b''.join([p[i] for i in range(n)])

        def _get_own_repr(self):
            if getattr(self, '_own', False):
                return 'owning %d bytes' % (ctypes.sizeof(self._as_ctype_ptr.contents),)
            return super(CTypesPtr, self)._get_own_repr()
    if BItem is self.ffi._get_cached_btype(model.void_type) or BItem is self.ffi._get_cached_btype(model.PrimitiveType('char')):
        CTypesPtr._automatic_casts = True
    CTypesPtr._fix_class()
    return CTypesPtr