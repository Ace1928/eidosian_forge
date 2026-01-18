import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def test_glyphs():
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 100, 20)
    context = Context(surface)
    font = context.get_scaled_font()
    text = 'Étt'
    glyphs, clusters, is_backwards = font.text_to_glyphs(5, 15, text, with_clusters=True)
    assert font.text_to_glyphs(5, 15, text, with_clusters=False) == glyphs
    (idx1, x1, y1), (idx2, x2, y2), (idx3, x3, y3) = glyphs
    assert idx1 != idx2 == idx3
    assert y1 == y2 == y3 == 15
    assert 5 == x1 < x2 < x3
    assert clusters == [(2, 1), (1, 1), (1, 1)]
    assert is_backwards == 0
    assert round_tuple(font.glyph_extents(glyphs)) == round_tuple(font.text_extents(text))
    assert round_tuple(font.glyph_extents(glyphs)) == round_tuple(context.glyph_extents(glyphs))
    assert context.copy_path() == []
    context.glyph_path(glyphs)
    glyph_path = context.copy_path()
    assert glyph_path
    context.new_path()
    assert context.copy_path() == []
    context.move_to(10, 20)
    context.text_path(text)
    assert context.copy_path() != []
    assert context.copy_path() != glyph_path
    context.new_path()
    assert context.copy_path() == []
    context.move_to(5, 15)
    context.text_path(text)
    text_path = context.copy_path()
    assert text_path[:-1] == glyph_path[:-1]
    empty = b'\x00' * 100 * 20 * 4
    assert surface.get_data()[:] == empty
    context.show_glyphs(glyphs)
    glyph_pixels = surface.get_data()[:]
    assert glyph_pixels != empty
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 100, 20)
    context = Context(surface)
    context.move_to(5, 15)
    context.show_text_glyphs(text, glyphs, clusters, is_backwards)
    text_glyphs_pixels = surface.get_data()[:]
    assert glyph_pixels == text_glyphs_pixels
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 100, 20)
    context = Context(surface)
    context.move_to(5, 15)
    context.show_text(text)
    text_pixels = surface.get_data()[:]
    assert glyph_pixels == text_pixels