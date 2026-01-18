import os
import sys
import weakref
from typing import Union, BinaryIO, Optional, Iterable
import pyglet
from pyglet.font.user import UserDefinedFontBase
from pyglet import gl
def add_user_font(font: UserDefinedFontBase):
    """Add a custom font created by the user.

    A strong reference needs to be applied to the font object,
    otherwise pyglet may not find the font later.

    :Parameters:
        `font` : `~pyglet.font.user.UserDefinedFont`
            A font class instance defined by user.
    """
    if not isinstance(font, UserDefinedFontBase):
        raise Exception('Font is not must be created fromm the UserDefinedFontBase.')
    shared_object_space = gl.current_context.object_space
    if not hasattr(shared_object_space, 'pyglet_font_font_cache'):
        shared_object_space.pyglet_font_font_cache = weakref.WeakValueDictionary()
        shared_object_space.pyglet_font_font_hold = []
        shared_object_space.pyglet_font_font_name_match = {}
    font_cache = shared_object_space.pyglet_font_font_cache
    font_hold = shared_object_space.pyglet_font_font_hold
    descriptor = (font.name, font.size, font.bold, font.italic, font.stretch, font.dpi)
    if descriptor in font_cache:
        raise Exception(f'A font with parameters {descriptor} has already been created.')
    if _system_font_class.have_font(font.name):
        raise Exception(f"Font name '{font.name}' already exists within the system fonts.")
    if font.name not in _user_fonts:
        _user_fonts.append(font.name)
    font_cache[descriptor] = font
    del font_hold[3:]
    font_hold.insert(0, font)