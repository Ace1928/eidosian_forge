from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
def element_id_attr(self):
    if self.element_id:
        return 'id="{element_id}"'.format(element_id=self.element_id)
    else:
        return ''