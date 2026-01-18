import os
import re
import shlex
import sys
import pygments
from pathlib import Path
from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
def slide(file_path, noclear=False, format_rst=True, formatter='terminal', style='native', auto_all=False, delimiter='...'):
    if noclear:
        demo_class = Demo
    else:
        demo_class = ClearDemo
    demo = demo_class(file_path, format_rst=format_rst, formatter=formatter, style=style, auto_all=auto_all)
    while not demo.finished:
        demo()
        try:
            py3compat.input('\n' + delimiter)
        except KeyboardInterrupt:
            exit(1)