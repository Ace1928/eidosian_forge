from __future__ import absolute_import, division, unicode_literals
from types import ModuleType
from six import text_type, PY3
def moduleFactoryFactory(factory):
    moduleCache = {}

    def moduleFactory(baseModule, *args, **kwargs):
        if isinstance(ModuleType.__name__, type('')):
            name = '_%s_factory' % baseModule.__name__
        else:
            name = b'_%s_factory' % baseModule.__name__
        kwargs_tuple = tuple(kwargs.items())
        try:
            return moduleCache[name][args][kwargs_tuple]
        except KeyError:
            mod = ModuleType(name)
            objs = factory(baseModule, *args, **kwargs)
            mod.__dict__.update(objs)
            if 'name' not in moduleCache:
                moduleCache[name] = {}
            if 'args' not in moduleCache[name]:
                moduleCache[name][args] = {}
            if 'kwargs' not in moduleCache[name][args]:
                moduleCache[name][args][kwargs_tuple] = {}
            moduleCache[name][args][kwargs_tuple] = mod
            return mod
    return moduleFactory