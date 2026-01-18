imported without importing all of the supporting library, so that we can
import importlib.util
import os
import sys
def load_file_as_module(name):
    path = os.path.join(os.path.dirname(bootstrap_file), '%s.py' % name)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod