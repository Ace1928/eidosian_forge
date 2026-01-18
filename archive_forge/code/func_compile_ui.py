from .Compiler import indenter, compiler
from .objcreator import widgetPluginPath
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