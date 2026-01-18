import ctypes, ctypes.util, operator, sys
from . import model
def new_array_type(self, CTypesPtr, length):
    if length is None:
        brackets = ' &[]'
    else:
        brackets = ' &[%d]' % length
    BItem = CTypesPtr._BItem
    getbtype = self.ffi._get_cached_btype
    if BItem is getbtype(model.PrimitiveType('char')):
        kind = 'char'
    elif BItem in (getbtype(model.PrimitiveType('signed char')), getbtype(model.PrimitiveType('unsigned char'))):
        kind = 'byte'
    else:
        kind = 'generic'

    class CTypesArray(CTypesGenericArray):
        __slots__ = ['_blob', '_own']
        if length is not None:
            _ctype = BItem._ctype * length
        else:
            __slots__.append('_ctype')
        _reftypename = BItem._get_c_name(brackets)
        _declared_length = length
        _CTPtr = CTypesPtr

        def __init__(self, init):
            if length is None:
                if isinstance(init, (int, long)):
                    len1 = init
                    init = None
                elif kind == 'char' and isinstance(init, bytes):
                    len1 = len(init) + 1
                else:
                    init = tuple(init)
                    len1 = len(init)
                self._ctype = BItem._ctype * len1
            self._blob = self._ctype()
            self._own = True
            if init is not None:
                self._initialize(self._blob, init)

        @staticmethod
        def _initialize(blob, init):
            if isinstance(init, bytes):
                init = [init[i:i + 1] for i in range(len(init))]
            else:
                if isinstance(init, CTypesGenericArray):
                    if len(init) != len(blob) or not isinstance(init, CTypesArray):
                        raise TypeError('length/type mismatch: %s' % (init,))
                init = tuple(init)
            if len(init) > len(blob):
                raise IndexError('too many initializers')
            addr = ctypes.cast(blob, ctypes.c_void_p).value
            PTR = ctypes.POINTER(BItem._ctype)
            itemsize = ctypes.sizeof(BItem._ctype)
            for i, value in enumerate(init):
                p = ctypes.cast(addr + i * itemsize, PTR)
                BItem._initialize(p.contents, value)

        def __len__(self):
            return len(self._blob)

        def __getitem__(self, index):
            if not 0 <= index < len(self._blob):
                raise IndexError
            return BItem._from_ctypes(self._blob[index])

        def __setitem__(self, index, value):
            if not 0 <= index < len(self._blob):
                raise IndexError
            self._blob[index] = BItem._to_ctypes(value)
        if kind == 'char' or kind == 'byte':

            def _to_string(self, maxlen):
                if maxlen < 0:
                    maxlen = len(self._blob)
                p = ctypes.cast(self._blob, ctypes.POINTER(ctypes.c_char))
                n = 0
                while n < maxlen and p[n] != b'\x00':
                    n += 1
                return b''.join([p[i] for i in range(n)])

        def _get_own_repr(self):
            if getattr(self, '_own', False):
                return 'owning %d bytes' % (ctypes.sizeof(self._blob),)
            return super(CTypesArray, self)._get_own_repr()

        def _convert_to_address(self, BClass):
            if BClass in (CTypesPtr, None) or BClass._automatic_casts:
                return ctypes.addressof(self._blob)
            else:
                return CTypesData._convert_to_address(self, BClass)

        @staticmethod
        def _from_ctypes(ctypes_array):
            self = CTypesArray.__new__(CTypesArray)
            self._blob = ctypes_array
            return self

        @staticmethod
        def _arg_to_ctypes(value):
            return CTypesPtr._arg_to_ctypes(value)

        def __add__(self, other):
            if isinstance(other, (int, long)):
                return CTypesPtr._new_pointer_at(ctypes.addressof(self._blob) + other * ctypes.sizeof(BItem._ctype))
            else:
                return NotImplemented

        @classmethod
        def _cast_from(cls, source):
            raise NotImplementedError('casting to %r' % (cls._get_c_name(),))
    CTypesArray._fix_class()
    return CTypesArray