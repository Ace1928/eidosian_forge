import shutil
import glob
import os
import sys
import tempfile
def handleReadonly(function, path, excinfo):
    excvalue = excinfo[1]
    if excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        function(path)
    else:
        raise