from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
@property
def foreground_true(self) -> bool:
    return self.__value & _FG_TRUE_COLOR != 0