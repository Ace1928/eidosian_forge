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
@property
def pipe_labels(self) -> Dict[str, List[str]]:
    """Get the labels set by the pipeline components, if available (if
        the component exposes a labels property and the labels are not
        hidden).

        RETURNS (Dict[str, List[str]]): Labels keyed by component name.
        """
    labels = {}
    for name, pipe in self._components:
        if hasattr(pipe, 'hide_labels') and pipe.hide_labels is True:
            continue
        if hasattr(pipe, 'labels'):
            labels[name] = list(pipe.labels)
    return SimpleFrozenDict(labels)