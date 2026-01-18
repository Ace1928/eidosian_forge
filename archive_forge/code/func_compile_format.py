import datetime
import functools
import logging
import os
import re
import time as time_mod
from collections import namedtuple
from typing import Any, Callable, Dict, Iterable, List, Tuple  # noqa
from .abc import AbstractAccessLogger
from .web_request import BaseRequest
from .web_response import StreamResponse
def compile_format(self, log_format: str) -> Tuple[str, List[KeyMethod]]:
    """Translate log_format into form usable by modulo formatting

        All known atoms will be replaced with %s
        Also methods for formatting of those atoms will be added to
        _methods in appropriate order

        For example we have log_format = "%a %t"
        This format will be translated to "%s %s"
        Also contents of _methods will be
        [self._format_a, self._format_t]
        These method will be called and results will be passed
        to translated string format.

        Each _format_* method receive 'args' which is list of arguments
        given to self.log

        Exceptions are _format_e, _format_i and _format_o methods which
        also receive key name (by functools.partial)

        """
    methods = list()
    for atom in self.FORMAT_RE.findall(log_format):
        if atom[1] == '':
            format_key1 = self.LOG_FORMAT_MAP[atom[0]]
            m = getattr(AccessLogger, '_format_%s' % atom[0])
            key_method = KeyMethod(format_key1, m)
        else:
            format_key2 = (self.LOG_FORMAT_MAP[atom[2]], atom[1])
            m = getattr(AccessLogger, '_format_%s' % atom[2])
            key_method = KeyMethod(format_key2, functools.partial(m, atom[1]))
        methods.append(key_method)
    log_format = self.FORMAT_RE.sub('%s', log_format)
    log_format = self.CLEANUP_RE.sub('%\\1', log_format)
    return (log_format, methods)