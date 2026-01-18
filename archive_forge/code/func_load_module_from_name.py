import os
import sys
import importlib.util as imputil
import mimetypes
from paste import request
from paste import fileapp
from paste.util import import_string
from paste import httpexceptions
from .httpheaders import ETAG
from paste.util import converters
def load_module_from_name(environ, filename, module_name, errors):
    if module_name in sys.modules:
        return sys.modules[module_name]
    init_filename = os.path.join(os.path.dirname(filename), '__init__.py')
    if not os.path.exists(init_filename):
        try:
            f = open(init_filename, 'w')
        except (OSError, IOError) as e:
            errors.write('Cannot write __init__.py file into directory %s (%s)\n' % (os.path.dirname(filename), e))
            return None
        f.write('#\n')
        f.close()
    if module_name in sys.modules:
        return sys.modules[module_name]
    if '.' in module_name:
        parent_name = '.'.join(module_name.split('.')[:-1])
        base_name = module_name.split('.')[-1]
        parent = load_module_from_name(environ, os.path.dirname(filename), parent_name, errors)
    else:
        base_name = module_name
    module = None
    spec = imputil.spec_from_file_location(base_name, filename)
    if spec is not None:
        module = imputil.module_from_spec(spec)
        sys.modules[base_name] = module
        spec.loader.exec_module(module)
    return module