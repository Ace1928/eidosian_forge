import _collections_abc
import sys as _sys
from itertools import chain as _chain
from itertools import repeat as _repeat
from itertools import starmap as _starmap
from keyword import iskeyword as _iskeyword
from operator import eq as _eq
from operator import itemgetter as _itemgetter
from reprlib import recursive_repr as _recursive_repr
from _weakref import proxy as _proxy
def namedtuple(typename, field_names, *, rename=False, defaults=None, module=None):
    """Returns a new subclass of tuple with named fields.

    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessible by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)

    """
    if isinstance(field_names, str):
        field_names = field_names.replace(',', ' ').split()
    field_names = list(map(str, field_names))
    typename = _sys.intern(str(typename))
    if rename:
        seen = set()
        for index, name in enumerate(field_names):
            if not name.isidentifier() or _iskeyword(name) or name.startswith('_') or (name in seen):
                field_names[index] = f'_{index}'
            seen.add(name)
    for name in [typename] + field_names:
        if type(name) is not str:
            raise TypeError('Type names and field names must be strings')
        if not name.isidentifier():
            raise ValueError(f'Type names and field names must be valid identifiers: {name!r}')
        if _iskeyword(name):
            raise ValueError(f'Type names and field names cannot be a keyword: {name!r}')
    seen = set()
    for name in field_names:
        if name.startswith('_') and (not rename):
            raise ValueError(f'Field names cannot start with an underscore: {name!r}')
        if name in seen:
            raise ValueError(f'Encountered duplicate field name: {name!r}')
        seen.add(name)
    field_defaults = {}
    if defaults is not None:
        defaults = tuple(defaults)
        if len(defaults) > len(field_names):
            raise TypeError('Got more default values than field names')
        field_defaults = dict(reversed(list(zip(reversed(field_names), reversed(defaults)))))
    field_names = tuple(map(_sys.intern, field_names))
    num_fields = len(field_names)
    arg_list = ', '.join(field_names)
    if num_fields == 1:
        arg_list += ','
    repr_fmt = '(' + ', '.join((f'{name}=%r' for name in field_names)) + ')'
    tuple_new = tuple.__new__
    _dict, _tuple, _len, _map, _zip = (dict, tuple, len, map, zip)
    namespace = {'_tuple_new': tuple_new, '__builtins__': {}, '__name__': f'namedtuple_{typename}'}
    code = f'lambda _cls, {arg_list}: _tuple_new(_cls, ({arg_list}))'
    __new__ = eval(code, namespace)
    __new__.__name__ = '__new__'
    __new__.__doc__ = f'Create new instance of {typename}({arg_list})'
    if defaults is not None:
        __new__.__defaults__ = defaults

    @classmethod
    def _make(cls, iterable):
        result = tuple_new(cls, iterable)
        if _len(result) != num_fields:
            raise TypeError(f'Expected {num_fields} arguments, got {len(result)}')
        return result
    _make.__func__.__doc__ = f'Make a new {typename} object from a sequence or iterable'

    def _replace(self, /, **kwds):
        result = self._make(_map(kwds.pop, field_names, self))
        if kwds:
            raise ValueError(f'Got unexpected field names: {list(kwds)!r}')
        return result
    _replace.__doc__ = f'Return a new {typename} object replacing specified fields with new values'

    def __repr__(self):
        """Return a nicely formatted representation string"""
        return self.__class__.__name__ + repr_fmt % self

    def _asdict(self):
        """Return a new dict which maps field names to their values."""
        return _dict(_zip(self._fields, self))

    def __getnewargs__(self):
        """Return self as a plain tuple.  Used by copy and pickle."""
        return _tuple(self)
    for method in (__new__, _make.__func__, _replace, __repr__, _asdict, __getnewargs__):
        method.__qualname__ = f'{typename}.{method.__name__}'
    class_namespace = {'__doc__': f'{typename}({arg_list})', '__slots__': (), '_fields': field_names, '_field_defaults': field_defaults, '__new__': __new__, '_make': _make, '_replace': _replace, '__repr__': __repr__, '_asdict': _asdict, '__getnewargs__': __getnewargs__, '__match_args__': field_names}
    for index, name in enumerate(field_names):
        doc = _sys.intern(f'Alias for field number {index}')
        class_namespace[name] = _tuplegetter(index, doc)
    result = type(typename, (tuple,), class_namespace)
    if module is None:
        try:
            module = _sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass
    if module is not None:
        result.__module__ = module
    return result