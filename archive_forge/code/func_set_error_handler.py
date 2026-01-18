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
def set_error_handler(self, error_handler: Callable[[str, PipeCallable, List[Doc], Exception], NoReturn]):
    """Set an error handler object for all the components in the pipeline
        that implement a set_error_handler function.

        error_handler (Callable[[str, Callable[[Doc], Doc], List[Doc], Exception], NoReturn]):
            Function that deals with a failing batch of documents. This callable
            function should take in the component's name, the component itself,
            the offending batch of documents, and the exception that was thrown.
        DOCS: https://spacy.io/api/language#set_error_handler
        """
    self.default_error_handler = error_handler
    for name, pipe in self.pipeline:
        if hasattr(pipe, 'set_error_handler'):
            pipe.set_error_handler(error_handler)