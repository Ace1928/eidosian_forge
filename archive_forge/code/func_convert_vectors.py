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
def convert_vectors(nlp: 'Language', vectors_loc: Optional[Path], *, truncate: int, prune: int, name: Optional[str]=None, mode: str=VectorsMode.default, attr: str='ORTH') -> None:
    vectors_loc = ensure_path(vectors_loc)
    if vectors_loc and vectors_loc.parts[-1].endswith('.npz'):
        if attr != 'ORTH':
            raise ValueError('ORTH is the only attribute supported for vectors in .npz format.')
        nlp.vocab.vectors = Vectors(strings=nlp.vocab.strings, data=numpy.load(vectors_loc.open('rb')))
        for lex in nlp.vocab:
            if lex.rank and lex.rank != OOV_RANK:
                nlp.vocab.vectors.add(lex.orth, row=lex.rank)
        nlp.vocab.deduplicate_vectors()
    else:
        if vectors_loc:
            logger.info('Reading vectors from %s', vectors_loc)
            vectors_data, vector_keys, floret_settings = read_vectors(vectors_loc, truncate, mode=mode)
            logger.info('Loaded vectors from %s', vectors_loc)
        else:
            vectors_data, vector_keys = (None, None)
        if vector_keys is not None and mode != VectorsMode.floret:
            for word in vector_keys:
                if word not in nlp.vocab:
                    nlp.vocab[word]
        if vectors_data is not None:
            if mode == VectorsMode.floret:
                nlp.vocab.vectors = Vectors(strings=nlp.vocab.strings, data=vectors_data, attr=attr, **floret_settings)
            else:
                nlp.vocab.vectors = Vectors(strings=nlp.vocab.strings, data=vectors_data, keys=vector_keys, attr=attr)
                nlp.vocab.deduplicate_vectors()
    if name is None:
        nlp.vocab.vectors.name = f'{nlp.meta['lang']}_{nlp.meta['name']}.vectors'
    else:
        nlp.vocab.vectors.name = name
    nlp.meta['vectors']['name'] = nlp.vocab.vectors.name
    if prune >= 1 and mode != VectorsMode.floret:
        nlp.vocab.prune_vectors(prune)