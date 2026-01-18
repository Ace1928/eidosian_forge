import os
import sys
import runpy
import types
from . import get_start_method, set_start_method
from . import process
from .context import reduction
from . import util
def get_preparation_data(name):
    """
    Return info about parent needed by child to unpickle process object
    """
    _check_not_importing_main()
    d = dict(log_to_stderr=util._log_to_stderr, authkey=process.current_process().authkey)
    if util._logger is not None:
        d['log_level'] = util._logger.getEffectiveLevel()
    sys_path = sys.path.copy()
    try:
        i = sys_path.index('')
    except ValueError:
        pass
    else:
        sys_path[i] = process.ORIGINAL_DIR
    d.update(name=name, sys_path=sys_path, sys_argv=sys.argv, orig_dir=process.ORIGINAL_DIR, dir=os.getcwd(), start_method=get_start_method())
    main_module = sys.modules['__main__']
    main_mod_name = getattr(main_module.__spec__, 'name', None)
    if main_mod_name is not None:
        d['init_main_from_name'] = main_mod_name
    elif sys.platform != 'win32' or (not WINEXE and (not WINSERVICE)):
        main_path = getattr(main_module, '__file__', None)
        if main_path is not None:
            if not os.path.isabs(main_path) and process.ORIGINAL_DIR is not None:
                main_path = os.path.join(process.ORIGINAL_DIR, main_path)
            d['init_main_from_path'] = os.path.normpath(main_path)
    return d