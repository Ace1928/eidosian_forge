import copyreg
import io
import functools
import types
import sys
import os
from multiprocessing import util
from pickle import loads, HIGHEST_PROTOCOL
def set_loky_pickler(loky_pickler=None):
    global _LokyPickler, _loky_pickler_name
    if loky_pickler is None:
        loky_pickler = ENV_LOKY_PICKLER
    loky_pickler_cls = None
    if loky_pickler in ['', None]:
        loky_pickler = 'cloudpickle'
    if loky_pickler == _loky_pickler_name:
        return
    if loky_pickler == 'cloudpickle':
        from joblib.externals.cloudpickle import CloudPickler as loky_pickler_cls
    else:
        try:
            from importlib import import_module
            module_pickle = import_module(loky_pickler)
            loky_pickler_cls = module_pickle.Pickler
        except (ImportError, AttributeError) as e:
            extra_info = f"\nThis error occurred while setting loky_pickler to '{loky_pickler}', as required by the env variable LOKY_PICKLER or the function set_loky_pickler."
            e.args = (e.args[0] + extra_info,) + e.args[1:]
            e.msg = e.args[0]
            raise e
    util.debug(f"Using '{(loky_pickler if loky_pickler else 'cloudpickle')}' for serialization.")

    class CustomizablePickler(loky_pickler_cls):
        _loky_pickler_cls = loky_pickler_cls

        def _set_dispatch_table(self, dispatch_table):
            for ancestor_class in self._loky_pickler_cls.mro():
                dt_attribute = getattr(ancestor_class, 'dispatch_table', None)
                if isinstance(dt_attribute, types.MemberDescriptorType):
                    dt_attribute.__set__(self, dispatch_table)
                    break
            self.dispatch_table = dispatch_table

        def __init__(self, writer, reducers=None, protocol=HIGHEST_PROTOCOL):
            loky_pickler_cls.__init__(self, writer, protocol=protocol)
            if reducers is None:
                reducers = {}
            if hasattr(self, 'dispatch_table'):
                loky_dt = dict(self.dispatch_table)
            else:
                loky_dt = copyreg.dispatch_table.copy()
            loky_dt.update(_dispatch_table)
            self._set_dispatch_table(loky_dt)
            for type, reduce_func in reducers.items():
                self.register(type, reduce_func)

        def register(self, type, reduce_func):
            """Attach a reducer function to a given type in the dispatch table."""
            self.dispatch_table[type] = reduce_func
    _LokyPickler = CustomizablePickler
    _loky_pickler_name = loky_pickler