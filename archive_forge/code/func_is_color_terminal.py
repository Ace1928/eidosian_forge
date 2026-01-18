import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
def is_color_terminal():
    """ Detect whether the environment supports color codes in output.

    .. versionadded:: 2.2.0
    """
    return (os.environ.get('WT_SESSION') or os.environ.get('COLORTERM') == 'truecolor' or os.environ.get('PYCHARM_HOSTED') == '1' or (os.environ.get('TERM') in ('rxvt', 'rxvt-256color', 'rxvt-unicode', 'rxvt-unicode-256color', 'xterm', 'xterm-256color'))) and platform not in ('android', 'ios')