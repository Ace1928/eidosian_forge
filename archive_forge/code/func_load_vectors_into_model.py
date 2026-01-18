import gzip
import tarfile
import warnings
import zipfile
from itertools import islice
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Dict, Optional, Union
import numpy
import srsly
import tqdm
from thinc.api import Config, ConfigValidationError, fix_random_seed, set_gpu_allocator
from ..errors import Errors, Warnings
from ..lookups import Lookups
from ..schemas import ConfigSchemaTraining
from ..util import (
from ..vectors import Mode as VectorsMode
from ..vectors import Vectors
from .pretrain import get_tok2vec_ref
def load_vectors_into_model(nlp: 'Language', name: Union[str, Path], *, add_strings: bool=True) -> None:
    """Load word vectors from an installed model or path into a model instance."""
    try:
        exclude = ['lookups']
        if not add_strings:
            exclude.append('strings')
        vectors_nlp = load_model(name, vocab=nlp.vocab, exclude=exclude)
    except ConfigValidationError as e:
        title = f'Config validation error for vectors {name}'
        desc = "This typically means that there's a problem in the config.cfg included with the packaged vectors. Make sure that the vectors package you're loading is compatible with the current version of spaCy."
        err = ConfigValidationError.from_error(e, title=title, desc=desc)
        raise err from None
    if len(vectors_nlp.vocab.vectors.keys()) == 0 and vectors_nlp.vocab.vectors.mode != VectorsMode.floret or (vectors_nlp.vocab.vectors.shape[0] == 0 and vectors_nlp.vocab.vectors.mode == VectorsMode.floret):
        logger.warning(Warnings.W112.format(name=name))
    for lex in nlp.vocab:
        lex.rank = nlp.vocab.vectors.key2row.get(lex.orth, OOV_RANK)