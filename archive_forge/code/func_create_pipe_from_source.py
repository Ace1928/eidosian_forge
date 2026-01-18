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
def create_pipe_from_source(self, source_name: str, source: 'Language', *, name: str) -> Tuple[PipeCallable, str]:
    """Create a pipeline component by copying it from an existing model.

        source_name (str): Name of the component in the source pipeline.
        source (Language): The source nlp object to copy from.
        name (str): Optional alternative name to use in current pipeline.
        RETURNS (Tuple[Callable[[Doc], Doc], str]): The component and its factory name.
        """
    if not isinstance(source, Language):
        raise ValueError(Errors.E945.format(name=source_name, source=type(source)))
    if self.vocab.vectors != source.vocab.vectors:
        warnings.warn(Warnings.W113.format(name=source_name))
    if source_name not in source.component_names:
        raise KeyError(Errors.E944.format(name=source_name, model=f'{source.meta['lang']}_{source.meta['name']}', opts=', '.join(source.component_names)))
    pipe = source.get_pipe(source_name)
    if hasattr(pipe, 'name'):
        pipe.name = name
    source_config = source.config.interpolate()
    pipe_config = util.copy_config(source_config['components'][source_name])
    self._pipe_configs[name] = pipe_config
    if self.vocab.strings != source.vocab.strings:
        for s in source.vocab.strings:
            self.vocab.strings.add(s)
    return (pipe, pipe_config['factory'])