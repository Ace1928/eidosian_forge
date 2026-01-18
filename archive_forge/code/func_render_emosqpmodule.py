from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def render_emosqpmodule(variables, output):
    """
    Render emosqpmodule.c file
    """
    python_ext_name = variables['python_ext_name']
    f = open(os.path.join(files_to_generate_path, 'emosqpmodule.c'))
    filedata = f.read()
    f.close()
    filedata = filedata.replace('PYTHON_EXT_NAME', str(python_ext_name))
    f = open(output, 'w')
    f.write(filedata)
    f.close()