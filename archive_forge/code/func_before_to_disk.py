import random
import shutil
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import (
from thinc.api import Config, Optimizer, constant, fix_random_seed, set_gpu_allocator
from wasabi import Printer
from ..errors import Errors
from ..schemas import ConfigSchemaTraining
from ..util import logger, registry, resolve_dot_names
from .example import Example
def before_to_disk(nlp: Language) -> Language:
    if not callback:
        return nlp
    modified_nlp = callback(nlp)
    if not isinstance(modified_nlp, Language):
        err = Errors.E914.format(name='before_to_disk', value=type(modified_nlp))
        raise ValueError(err)
    return modified_nlp