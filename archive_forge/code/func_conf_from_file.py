import re
import inspect
import os
import sys
from importlib.machinery import SourceFileLoader
def conf_from_file(filepath):
    """
    Creates a configuration dictionary from a file.

    :param filepath: The path to the file.
    """
    abspath = os.path.abspath(os.path.expanduser(filepath))
    conf_dict = {}
    if not os.path.isfile(abspath):
        raise RuntimeError('`%s` is not a file.' % abspath)
    with open(abspath, 'rb') as f:
        compiled = compile(f.read(), abspath, 'exec')
    absname, _ = os.path.splitext(abspath)
    basepath, module_name = absname.rsplit(os.sep, 1)
    SourceFileLoader(module_name, abspath).load_module(module_name)
    exec(compiled, globals(), conf_dict)
    conf_dict['__file__'] = abspath
    return conf_from_dict(conf_dict)