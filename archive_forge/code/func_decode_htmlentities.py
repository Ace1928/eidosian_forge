from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types
import numpy as np
import scipy.sparse
from smart_open import open
from gensim import __version__ as gensim_version
def decode_htmlentities(text):
    """Decode all HTML entities in text that are encoded as hex, decimal or named entities.
    Adapted from `python-twitter-ircbot/html_decode.py
    <https://github.com/sku/python-twitter-ircbot/blob/321d94e0e40d0acc92f5bf57d126b57369da70de/html_decode.py>`_.

    Parameters
    ----------
    text : str
        Input HTML.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import decode_htmlentities
        >>>
        >>> u = u'E tu vivrai nel terrore - L&#x27;aldil&#xE0; (1981)'
        >>> print(decode_htmlentities(u).encode('UTF-8'))
        E tu vivrai nel terrore - L'aldilà (1981)
        >>> print(decode_htmlentities("l&#39;eau"))
        l'eau
        >>> print(decode_htmlentities("foo &lt; bar"))
        foo < bar

    """

    def substitute_entity(match):
        try:
            ent = match.group(3)
            if match.group(1) == '#':
                if match.group(2) == '':
                    return safe_unichr(int(ent))
                elif match.group(2) in ['x', 'X']:
                    return safe_unichr(int(ent, 16))
            else:
                cp = n2cp.get(ent)
                if cp:
                    return safe_unichr(cp)
                else:
                    return match.group()
        except Exception:
            return match.group()
    return RE_HTML_ENTITY.sub(substitute_entity, text)