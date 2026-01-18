import ctypes, ctypes.util, operator, sys
from . import model
def new_enum_type(self, name, enumerators, enumvalues, CTypesInt):
    assert isinstance(name, str)
    reverse_mapping = dict(zip(reversed(enumvalues), reversed(enumerators)))

    class CTypesEnum(CTypesInt):
        __slots__ = []
        _reftypename = '%s &' % name

        def _get_own_repr(self):
            value = self._value
            try:
                return '%d: %s' % (value, reverse_mapping[value])
            except KeyError:
                return str(value)

        def _to_string(self, maxlen):
            value = self._value
            try:
                return reverse_mapping[value]
            except KeyError:
                return str(value)
    CTypesEnum._fix_class()
    return CTypesEnum