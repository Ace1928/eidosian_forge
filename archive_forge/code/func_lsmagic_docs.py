import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
def lsmagic_docs(self, brief=False, missing=''):
    """Return dict of documentation of magic functions.

        The return dict has the keys 'line' and 'cell', corresponding to the
        two types of magics we support. Each value is a dict keyed by magic
        name whose value is the function docstring. If a docstring is
        unavailable, the value of `missing` is used instead.

        If brief is True, only the first line of each docstring will be returned.
        """
    docs = {}
    for m_type in self.magics:
        m_docs = {}
        for m_name, m_func in self.magics[m_type].items():
            if m_func.__doc__:
                if brief:
                    m_docs[m_name] = m_func.__doc__.split('\n', 1)[0]
                else:
                    m_docs[m_name] = m_func.__doc__.rstrip()
            else:
                m_docs[m_name] = missing
        docs[m_type] = m_docs
    return docs