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
def get_cached_extents(self):
    """Returns a cached version of the :meth:`get_extents` function.

        ::

            >>> func = self._get_cached_extents()
            >>> func
            <built-in method size of pygame.font.Font object at 0x01E45650>
            >>> func('a line')
            (36, 18)

        .. warning::

            This method returns a size measuring function that is valid
            for the font settings used at the time :meth:`get_cached_extents`
            was called. Any change in the font settings will render the
            returned function incorrect. You should only use this if you know
            what you're doing.

        .. versionadded:: 1.9.0
        """
    return self.get_extents