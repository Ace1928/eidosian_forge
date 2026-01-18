from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
def save_reduce(self, func, args, state=None, listitems=None, dictitems=None, state_setter=None, *, obj=None):
    if not isinstance(args, tuple):
        raise PicklingError('args from save_reduce() must be a tuple')
    if not callable(func):
        raise PicklingError('func from save_reduce() must be callable')
    save = self.save
    write = self.write
    func_name = getattr(func, '__name__', '')
    if self.proto >= 2 and func_name == '__newobj_ex__':
        cls, args, kwargs = args
        if not hasattr(cls, '__new__'):
            raise PicklingError('args[0] from {} args has no __new__'.format(func_name))
        if obj is not None and cls is not obj.__class__:
            raise PicklingError('args[0] from {} args has the wrong class'.format(func_name))
        if self.proto >= 4:
            save(cls)
            save(args)
            save(kwargs)
            write(NEWOBJ_EX)
        else:
            func = partial(cls.__new__, cls, *args, **kwargs)
            save(func)
            save(())
            write(REDUCE)
    elif self.proto >= 2 and func_name == '__newobj__':
        cls = args[0]
        if not hasattr(cls, '__new__'):
            raise PicklingError('args[0] from __newobj__ args has no __new__')
        if obj is not None and cls is not obj.__class__:
            raise PicklingError('args[0] from __newobj__ args has the wrong class')
        args = args[1:]
        save(cls)
        save(args)
        write(NEWOBJ)
    else:
        save(func)
        save(args)
        write(REDUCE)
    if obj is not None:
        if id(obj) in self.memo:
            write(POP + self.get(self.memo[id(obj)][0]))
        else:
            self.memoize(obj)
    if listitems is not None:
        self._batch_appends(listitems)
    if dictitems is not None:
        self._batch_setitems(dictitems)
    if state is not None:
        if state_setter is None:
            save(state)
            write(BUILD)
        else:
            save(state_setter)
            save(obj)
            save(state)
            write(TUPLE2)
            write(REDUCE)
            write(POP)