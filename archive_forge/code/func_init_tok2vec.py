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
def init_tok2vec(nlp: 'Language', pretrain_config: Dict[str, Any], init_config: Dict[str, Any]) -> bool:
    P = pretrain_config
    I = init_config
    weights_data = None
    init_tok2vec = ensure_path(I['init_tok2vec'])
    if init_tok2vec is not None:
        if not init_tok2vec.exists():
            err = f"can't find pretrained tok2vec: {init_tok2vec}"
            errors = [{'loc': ['initialize', 'init_tok2vec'], 'msg': err}]
            raise ConfigValidationError(config=nlp.config, errors=errors)
        with init_tok2vec.open('rb') as file_:
            weights_data = file_.read()
    if weights_data is not None:
        layer = get_tok2vec_ref(nlp, P)
        layer.from_bytes(weights_data)
        logger.info('Loaded pretrained weights from %s', init_tok2vec)
        return True
    return False