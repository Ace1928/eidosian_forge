from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
def autoplay_attr(self):
    if self.autoplay:
        return 'autoplay="autoplay"'
    else:
        return ''