import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
def parseModes(modes, params, paramModes=('', '')):
    """
    Parse an IRC mode string.

    The mode string is parsed into two lists of mode changes (added and
    removed), with each mode change represented as C{(mode, param)} where mode
    is the mode character, and param is the parameter passed for that mode, or
    L{None} if no parameter is required.

    @type modes: C{str}
    @param modes: Modes string to parse.

    @type params: C{list}
    @param params: Parameters specified along with L{modes}.

    @type paramModes: C{(str, str)}
    @param paramModes: A pair of strings (C{(add, remove)}) that indicate which modes take
        parameters when added or removed.

    @returns: Two lists of mode changes, one for modes added and the other for
        modes removed respectively, mode changes in each list are represented as
        C{(mode, param)}.
    """
    if len(modes) == 0:
        raise IRCBadModes('Empty mode string')
    if modes[0] not in '+-':
        raise IRCBadModes(f'Malformed modes string: {modes!r}')
    changes = ([], [])
    direction = None
    count = -1
    for ch in modes:
        if ch in '+-':
            if count == 0:
                raise IRCBadModes(f'Empty mode sequence: {modes!r}')
            direction = '+-'.index(ch)
            count = 0
        else:
            param = None
            if ch in paramModes[direction]:
                try:
                    param = params.pop(0)
                except IndexError:
                    raise IRCBadModes(f'Not enough parameters: {ch!r}')
            changes[direction].append((ch, param))
            count += 1
    if len(params) > 0:
        raise IRCBadModes(f'Too many parameters: {modes!r} {params!r}')
    if count == 0:
        raise IRCBadModes(f'Empty mode sequence: {modes!r}')
    return changes