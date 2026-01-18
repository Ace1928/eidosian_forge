from .Compiler import indenter, compiler
from .objcreator import widgetPluginPath
def compileUiDir(dir, recurse=False, map=None, **compileUi_args):
    """compileUiDir(dir, recurse=False, map=None, **compileUi_args)

    Creates Python modules from Qt Designer .ui files in a directory or
    directory tree.

    dir is the name of the directory to scan for files whose name ends with
    '.ui'.  By default the generated Python module is created in the same
    directory ending with '.py'.
    recurse is set if any sub-directories should be scanned.  The default is
    False.
    map is an optional callable that is passed the name of the directory
    containing the '.ui' file and the name of the Python module that will be
    created.  The callable should return a tuple of the name of the directory
    in which the Python module will be created and the (possibly modified)
    name of the module.  The default is None.
    compileUi_args are any additional keyword arguments that are passed to
    the compileUi() function that is called to create each Python module.
    """
    import os

    def compile_ui(ui_dir, ui_file):
        if ui_file.endswith('.ui'):
            py_dir = ui_dir
            py_file = ui_file[:-3] + '.py'
            if map is not None:
                py_dir, py_file = map(py_dir, py_file)
            try:
                os.makedirs(py_dir)
            except:
                pass
            ui_path = os.path.join(ui_dir, ui_file)
            py_path = os.path.join(py_dir, py_file)
            try:
                py_file = open(py_path, 'w', encoding='utf-8')
            except TypeError:
                py_file = open(py_path, 'w')
            try:
                compileUi(ui_path, py_file, **compileUi_args)
            finally:
                py_file.close()
    if recurse:
        for root, _, files in os.walk(dir):
            for ui in files:
                compile_ui(root, ui)
    else:
        for ui in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, ui)):
                compile_ui(dir, ui)