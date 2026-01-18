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
def add_pipe(self, factory_name: str, name: Optional[str]=None, *, before: Optional[Union[str, int]]=None, after: Optional[Union[str, int]]=None, first: Optional[bool]=None, last: Optional[bool]=None, source: Optional['Language']=None, config: Dict[str, Any]=SimpleFrozenDict(), raw_config: Optional[Config]=None, validate: bool=True) -> PipeCallable:
    """Add a component to the processing pipeline. Valid components are
        callables that take a `Doc` object, modify it and return it. Only one
        of before/after/first/last can be set. Default behaviour is "last".

        factory_name (str): Name of the component factory.
        name (str): Name of pipeline component. Overwrites existing
            component.name attribute if available. If no name is set and
            the component exposes no name attribute, component.__name__ is
            used. An error is raised if a name already exists in the pipeline.
        before (Union[str, int]): Name or index of the component to insert new
            component directly before.
        after (Union[str, int]): Name or index of the component to insert new
            component directly after.
        first (bool): If True, insert component first in the pipeline.
        last (bool): If True, insert component last in the pipeline.
        source (Language): Optional loaded nlp object to copy the pipeline
            component from.
        config (Dict[str, Any]): Config parameters to use for this component.
            Will be merged with default config, if available.
        raw_config (Optional[Config]): Internals: the non-interpolated config.
        validate (bool): Whether to validate the component config against the
            arguments and types expected by the factory.
        RETURNS (Callable[[Doc], Doc]): The pipeline component.

        DOCS: https://spacy.io/api/language#add_pipe
        """
    if not isinstance(factory_name, str):
        bad_val = repr(factory_name)
        err = Errors.E966.format(component=bad_val, name=name)
        raise ValueError(err)
    name = name if name is not None else factory_name
    if name in self.component_names:
        raise ValueError(Errors.E007.format(name=name, opts=self.component_names))
    if 'name' in config:
        warnings.warn(Warnings.W119.format(name_in_config=config.pop('name')))
    if source is not None:
        pipe_component, factory_name = self.create_pipe_from_source(factory_name, source, name=name)
    else:
        pipe_component = self.create_pipe(factory_name, name=name, config=config, raw_config=raw_config, validate=validate)
    pipe_index = self._get_pipe_index(before, after, first, last)
    self._pipe_meta[name] = self.get_factory_meta(factory_name)
    self._components.insert(pipe_index, (name, pipe_component))
    self._link_components()
    return pipe_component