from __future__ import annotations
import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple
import numpy as np
from pyparsing import (
import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, KERNING_DEFAULT
from packaging.version import parse as parse_version
from pyparsing import __version__ as pyparsing_version
def ship(box: Box, xy: tuple[float, float]=(0, 0)) -> Output:
    """
    Ship out *box* at offset *xy*, converting it to an `Output`.

    Since boxes can be inside of boxes inside of boxes, the main work of `ship`
    is done by two mutually recursive routines, `hlist_out` and `vlist_out`,
    which traverse the `Hlist` nodes and `Vlist` nodes inside of horizontal
    and vertical boxes.  The global variables used in TeX to store state as it
    processes have become local variables here.
    """
    ox, oy = xy
    cur_v = 0.0
    cur_h = 0.0
    off_h = ox
    off_v = oy + box.height
    output = Output(box)

    def clamp(value: float) -> float:
        return -1000000000.0 if value < -1000000000.0 else +1000000000.0 if value > +1000000000.0 else value

    def hlist_out(box: Hlist) -> None:
        nonlocal cur_v, cur_h, off_h, off_v
        cur_g = 0
        cur_glue = 0.0
        glue_order = box.glue_order
        glue_sign = box.glue_sign
        base_line = cur_v
        left_edge = cur_h
        for p in box.children:
            if isinstance(p, Char):
                p.render(output, cur_h + off_h, cur_v + off_v)
                cur_h += p.width
            elif isinstance(p, Kern):
                cur_h += p.width
            elif isinstance(p, List):
                if len(p.children) == 0:
                    cur_h += p.width
                else:
                    edge = cur_h
                    cur_v = base_line + p.shift_amount
                    if isinstance(p, Hlist):
                        hlist_out(p)
                    elif isinstance(p, Vlist):
                        vlist_out(p)
                    else:
                        assert False, 'unreachable code'
                    cur_h = edge + p.width
                    cur_v = base_line
            elif isinstance(p, Box):
                rule_height = p.height
                rule_depth = p.depth
                rule_width = p.width
                if np.isinf(rule_height):
                    rule_height = box.height
                if np.isinf(rule_depth):
                    rule_depth = box.depth
                if rule_height > 0 and rule_width > 0:
                    cur_v = base_line + rule_depth
                    p.render(output, cur_h + off_h, cur_v + off_v, rule_width, rule_height)
                    cur_v = base_line
                cur_h += rule_width
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                rule_width = glue_spec.width - cur_g
                if glue_sign != 0:
                    if glue_sign == 1:
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_width += cur_g
                cur_h += rule_width

    def vlist_out(box: Vlist) -> None:
        nonlocal cur_v, cur_h, off_h, off_v
        cur_g = 0
        cur_glue = 0.0
        glue_order = box.glue_order
        glue_sign = box.glue_sign
        left_edge = cur_h
        cur_v -= box.height
        top_edge = cur_v
        for p in box.children:
            if isinstance(p, Kern):
                cur_v += p.width
            elif isinstance(p, List):
                if len(p.children) == 0:
                    cur_v += p.height + p.depth
                else:
                    cur_v += p.height
                    cur_h = left_edge + p.shift_amount
                    save_v = cur_v
                    p.width = box.width
                    if isinstance(p, Hlist):
                        hlist_out(p)
                    elif isinstance(p, Vlist):
                        vlist_out(p)
                    else:
                        assert False, 'unreachable code'
                    cur_v = save_v + p.depth
                    cur_h = left_edge
            elif isinstance(p, Box):
                rule_height = p.height
                rule_depth = p.depth
                rule_width = p.width
                if np.isinf(rule_width):
                    rule_width = box.width
                rule_height += rule_depth
                if rule_height > 0 and rule_depth > 0:
                    cur_v += rule_height
                    p.render(output, cur_h + off_h, cur_v + off_v, rule_width, rule_height)
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                rule_height = glue_spec.width - cur_g
                if glue_sign != 0:
                    if glue_sign == 1:
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_height += cur_g
                cur_v += rule_height
            elif isinstance(p, Char):
                raise RuntimeError('Internal mathtext error: Char node found in vlist')
    assert isinstance(box, Hlist)
    hlist_out(box)
    return output