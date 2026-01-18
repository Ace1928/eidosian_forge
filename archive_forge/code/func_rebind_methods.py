import types
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_trace_api
def rebind_methods(self):
    if len(self.active_plugins) == 0:
        self.bind_functions(pydevd_trace_api, getattr, pydevd_trace_api)
    elif len(self.active_plugins) == 1:
        self.bind_functions(pydevd_trace_api, getattr, self.active_plugins[0])
    else:
        self.bind_functions(pydevd_trace_api, create_dispatch, self.active_plugins)