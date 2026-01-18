from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def render_setuppy(variables, output):
    """
    Render setup.py file
    """
    embedded_flag = variables['embedded_flag']
    python_ext_name = variables['python_ext_name']
    f = open(os.path.join(files_to_generate_path, 'setup.py'))
    filedata = f.read()
    f.close()
    filedata = filedata.replace('EMBEDDED_FLAG', str(embedded_flag))
    filedata = filedata.replace('PYTHON_EXT_NAME', str(python_ext_name))
    f = open(output, 'w')
    f.write(filedata)
    f.close()