import os
from os.path import join
import shutil
import tempfile
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import debug
from .latex import latex
def on_expose():
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    img.blit((win.width - img.width) / 2, (win.height - img.height) / 2)