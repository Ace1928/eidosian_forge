from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def render_cmakelists(variables, output):
    """
    Render CMakeLists file
    """
    embedded_flag = variables['embedded_flag']
    f = open(os.path.join(files_to_generate_path, 'CMakeLists.txt'))
    filedata = f.read()
    f.close()
    filedata = filedata.replace('EMBEDDED_FLAG', str(embedded_flag))
    f = open(output, 'w')
    f.write(filedata)
    f.close()