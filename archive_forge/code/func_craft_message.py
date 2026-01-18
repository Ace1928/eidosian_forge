from __future__ import annotations
import functools
import logging
import os
import subprocess
import sys
import warnings
from datetime import datetime
from typing import TYPE_CHECKING
def craft_message(old: Callable, replacement: Callable, message: str, deadline: datetime) -> str:
    msg = f'{old.__name__} is deprecated'
    if deadline is not None:
        msg += f', and will be removed on {_deadline:%Y-%m-%d}\n'
    if replacement is not None:
        if isinstance(replacement, property):
            r = replacement.fget
        elif isinstance(replacement, (classmethod, staticmethod)):
            r = replacement.__func__
        else:
            r = replacement
        msg += f'; use {r.__name__} in {r.__module__} instead.'
    if message:
        msg += '\n' + message
    return msg