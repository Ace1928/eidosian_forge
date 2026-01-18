import os
from os.path import join
import shutil
import tempfile
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import debug
from .latex import latex
def system_default_viewer(fname, fmt):
    """ Open fname with the default system viewer.

    In practice, it is impossible for python to know when the system viewer is
    done. For this reason, we ensure the passed file will not be deleted under
    it, and this function does not attempt to block.
    """
    with tempfile.NamedTemporaryFile(prefix='sympy-preview-', suffix=os.path.splitext(fname)[1], delete=False) as temp_f:
        with open(fname, 'rb') as f:
            shutil.copyfileobj(f, temp_f)
    import platform
    if platform.system() == 'Darwin':
        import subprocess
        subprocess.call(('open', temp_f.name))
    elif platform.system() == 'Windows':
        os.startfile(temp_f.name)
    else:
        import subprocess
        subprocess.call(('xdg-open', temp_f.name))