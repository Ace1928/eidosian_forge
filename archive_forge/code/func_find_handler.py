from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
@cython.final
def find_handler(self, obj):
    cls = type(obj)
    mro = inspect.getmro(cls)
    for mro_cls in mro:
        handler_method = getattr(self, 'visit_' + mro_cls.__name__, None)
        if handler_method is not None:
            return handler_method
    print(type(self), cls)
    if self.access_path:
        print(self.access_path)
        print(self.access_path[-1][0].pos)
        print(self.access_path[-1][0].__dict__)
    raise RuntimeError('Visitor %r does not accept object: %s' % (self, obj))