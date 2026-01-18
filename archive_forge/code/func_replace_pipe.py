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
def replace_pipe(self, name: str, factory_name: str, *, config: Dict[str, Any]=SimpleFrozenDict(), validate: bool=True) -> PipeCallable:
    """Replace a component in the pipeline.

        name (str): Name of the component to replace.
        factory_name (str): Factory name of replacement component.
        config (Optional[Dict[str, Any]]): Config parameters to use for this
            component. Will be merged with default config, if available.
        validate (bool): Whether to validate the component config against the
            arguments and types expected by the factory.
        RETURNS (Callable[[Doc], Doc]): The new pipeline component.

        DOCS: https://spacy.io/api/language#replace_pipe
        """
    if name not in self.component_names:
        raise ValueError(Errors.E001.format(name=name, opts=self.pipe_names))
    if hasattr(factory_name, '__call__'):
        err = Errors.E968.format(component=repr(factory_name), name=name)
        raise ValueError(err)
    pipe_index = self.component_names.index(name)
    self.remove_pipe(name)
    if not len(self._components) or pipe_index == len(self._components):
        return self.add_pipe(factory_name, name=name, config=config, validate=validate)
    else:
        return self.add_pipe(factory_name, name=name, before=pipe_index, config=config, validate=validate)