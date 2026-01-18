import os
import subprocess as sp
from .compat import DEVNULL
from .config_defaults import FFMPEG_BINARY, IMAGEMAGICK_BINARY
def change_settings(new_settings=None, filename=None):
    """ Changes the value of configuration variables."""
    new_settings = new_settings or {}
    gl = globals()
    if filename:
        with open(filename) as in_file:
            exec(in_file)
        gl.update(locals())
    gl.update(new_settings)