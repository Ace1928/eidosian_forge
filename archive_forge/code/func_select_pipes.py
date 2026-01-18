import functools
import inspect
import itertools
import multiprocessing as mp
import random
import traceback
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain, cycle
from pathlib import Path
from timeit import default_timer as timer
from typing import (
import srsly
from thinc.api import Config, CupyOps, Optimizer, get_current_ops
from . import about, ty, util
from .compat import Literal
from .errors import Errors, Warnings
from .git_info import GIT_VERSION
from .lang.punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .lang.tokenizer_exceptions import BASE_EXCEPTIONS, URL_MATCH
from .lookups import load_lookups
from .pipe_analysis import analyze_pipes, print_pipe_analysis, validate_attrs
from .schemas import (
from .scorer import Scorer
from .tokenizer import Tokenizer
from .tokens import Doc
from .tokens.underscore import Underscore
from .training import Example, validate_examples
from .training.initialize import init_tok2vec, init_vocab
from .util import (
from .vectors import BaseVectors
from .vocab import Vocab, create_vocab
def select_pipes(self, *, disable: Optional[Union[str, Iterable[str]]]=None, enable: Optional[Union[str, Iterable[str]]]=None) -> 'DisabledPipes':
    """Disable one or more pipeline components. If used as a context
        manager, the pipeline will be restored to the initial state at the end
        of the block. Otherwise, a DisabledPipes object is returned, that has
        a `.restore()` method you can use to undo your changes.

        disable (str or iterable): The name(s) of the pipes to disable
        enable (str or iterable): The name(s) of the pipes to enable - all others will be disabled

        DOCS: https://spacy.io/api/language#select_pipes
        """
    if enable is None and disable is None:
        raise ValueError(Errors.E991)
    if isinstance(disable, str):
        disable = [disable]
    if enable is not None:
        if isinstance(enable, str):
            enable = [enable]
        to_disable = [pipe for pipe in self.pipe_names if pipe not in enable]
        if disable is not None and disable != to_disable:
            raise ValueError(Errors.E992.format(enable=enable, disable=disable, names=self.pipe_names))
        disable = to_disable
    assert disable is not None
    disable = [d for d in disable if d not in self._disabled]
    return DisabledPipes(self, disable)