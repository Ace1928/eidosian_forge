importers when locating support scripts as well as when importing modules.
import sys
import importlib.machinery # importlib first so we can test #15386 via -m
import importlib.util
import io
import os
def run_path(path_name, init_globals=None, run_name=None):
    """Execute code located at the specified filesystem location.

       path_name -- filesystem location of a Python script, zipfile,
       or directory containing a top level __main__.py script.

       Optional arguments:
       init_globals -- dictionary used to pre-populate the module’s
       globals dictionary before the code is executed.

       run_name -- if not None, this will be used to set __name__;
       otherwise, '<run_path>' will be used for __name__.

       Returns the resulting module globals dictionary.
    """
    if run_name is None:
        run_name = '<run_path>'
    pkg_name = run_name.rpartition('.')[0]
    from pkgutil import get_importer
    importer = get_importer(path_name)
    is_NullImporter = False
    if type(importer).__module__ == 'imp':
        if type(importer).__name__ == 'NullImporter':
            is_NullImporter = True
    if isinstance(importer, type(None)) or is_NullImporter:
        code, fname = _get_code_from_file(run_name, path_name)
        return _run_module_code(code, init_globals, run_name, pkg_name=pkg_name, script_name=fname)
    else:
        sys.path.insert(0, path_name)
        try:
            mod_name, mod_spec, code = _get_main_module_details()
            with _TempModule(run_name) as temp_module, _ModifiedArgv0(path_name):
                mod_globals = temp_module.module.__dict__
                return _run_code(code, mod_globals, init_globals, run_name, mod_spec, pkg_name).copy()
        finally:
            try:
                sys.path.remove(path_name)
            except ValueError:
                pass