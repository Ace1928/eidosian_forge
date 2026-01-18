import re
import os
from ast import literal_eval
from functools import partial
from copy import copy
from kivy import kivy_data_dir
from kivy.config import Config
from kivy.utils import platform
from kivy.graphics.texture import Texture
from kivy.core import core_select_lib
from kivy.core.text.text_layout import layout_text, LayoutWord
from kivy.resources import resource_find, resource_add_path
from kivy.compat import PY2
from kivy.setupconfig import USE_SDL2, USE_PANGOFT2
from kivy.logger import Logger
@staticmethod
def list_families(font_context):
    """Returns a list of `bytes` objects, each representing a font family
        name that is available in the given `font_context`.

        .. versionadded:: 1.11.0

        .. note::
            Pango adds static "Serif", "Sans" and "Monospace" to the list in
            current versions, even if only a single custom font file is added
            to the context.

        .. note:: This feature requires the Pango text provider.
        """
    raise NotImplementedError('No font_context support in text provider')