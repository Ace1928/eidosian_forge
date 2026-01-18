from __future__ import annotations
import asyncio
import codecs
import itertools
import logging
import os
import select
import signal
import warnings
from collections import deque
from concurrent import futures
from typing import TYPE_CHECKING, Any, Coroutine
from tornado.ioloop import IOLoop
def make_term_env(self, height: int=25, width: int=80, winheight: int=0, winwidth: int=0, **kwargs: Any) -> dict[str, str]:
    """Build the environment variables for the process in the terminal."""
    env = os.environ.copy()
    env['TERM'] = self.term_settings.get('type', DEFAULT_TERM_TYPE)
    dimensions = '%dx%d' % (width, height)
    if winwidth and winheight:
        dimensions += ';%dx%d' % (winwidth, winheight)
    env[ENV_PREFIX + 'DIMENSIONS'] = dimensions
    env['COLUMNS'] = str(width)
    env['LINES'] = str(height)
    if self.server_url:
        env[ENV_PREFIX + 'URL'] = self.server_url
    if self.extra_env:
        _update_removing(env, self.extra_env)
    term_env = kwargs.get('extra_env', {})
    if term_env and isinstance(term_env, dict):
        _update_removing(env, term_env)
    return env