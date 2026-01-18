import os
import subprocess as sp
from .compat import DEVNULL
from .config_defaults import FFMPEG_BINARY, IMAGEMAGICK_BINARY
def get_setting(varname):
    """ Returns the value of a configuration variable. """
    gl = globals()
    if varname not in gl.keys():
        raise ValueError('Unknown setting %s' % varname)
    return gl[varname]