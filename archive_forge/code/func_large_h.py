from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def large_h(desc: str) -> bool:
    if not desc.startswith('h'):
        return False
    if ',' in desc:
        desc = desc.split(',', 1)[0]
    num = int(desc[1:], 10)
    return num > 15