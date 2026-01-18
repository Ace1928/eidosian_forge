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
def init_nlp(config: Config, *, use_gpu: int=-1) -> 'Language':
    raw_config = config
    config = raw_config.interpolate()
    if 'seed' not in config['training']:
        raise ValueError(Errors.E1015.format(value='[training] seed'))
    if 'gpu_allocator' not in config['training']:
        raise ValueError(Errors.E1015.format(value='[training] gpu_allocator'))
    if config['training']['seed'] is not None:
        fix_random_seed(config['training']['seed'])
    allocator = config['training']['gpu_allocator']
    if use_gpu >= 0 and allocator:
        set_gpu_allocator(allocator)
    sourced = get_sourced_components(config)
    nlp = load_model_from_config(raw_config, auto_fill=True)
    logger.info('Set up nlp object from config')
    config = nlp.config.interpolate()
    T = registry.resolve(config['training'], schema=ConfigSchemaTraining)
    dot_names = [T['train_corpus'], T['dev_corpus']]
    if not isinstance(T['train_corpus'], str):
        raise ConfigValidationError(desc=Errors.E897.format(field='training.train_corpus', type=type(T['train_corpus'])))
    if not isinstance(T['dev_corpus'], str):
        raise ConfigValidationError(desc=Errors.E897.format(field='training.dev_corpus', type=type(T['dev_corpus'])))
    train_corpus, dev_corpus = resolve_dot_names(config, dot_names)
    optimizer = T['optimizer']
    frozen_components = T['frozen_components']
    resume_components = [p for p in sourced if p not in frozen_components]
    logger.info('Pipeline: %s', nlp.pipe_names)
    if resume_components:
        with nlp.select_pipes(enable=resume_components):
            logger.info('Resuming training for: %s', resume_components)
            nlp.resume_training(sgd=optimizer)
    nlp._link_components()
    with nlp.select_pipes(disable=[*frozen_components, *resume_components]):
        if T['max_epochs'] == -1:
            sample_size = 100
            logger.debug('Due to streamed train corpus, using only first %s examples for initialization. If necessary, provide all labels in [initialize]. More info: https://spacy.io/api/cli#init_labels', sample_size)
            nlp.initialize(lambda: islice(train_corpus(nlp), sample_size), sgd=optimizer)
        else:
            nlp.initialize(lambda: train_corpus(nlp), sgd=optimizer)
        logger.info('Initialized pipeline components: %s', nlp.pipe_names)
    for name, proc in nlp.pipeline:
        for listener in getattr(proc, 'listening_components', []):
            if listener not in nlp.pipe_names:
                continue
            if listener in frozen_components and name not in frozen_components:
                logger.warning(Warnings.W087.format(name=name, listener=listener))
            if listener not in frozen_components and name in frozen_components:
                if name not in T['annotating_components']:
                    logger.warning(Warnings.W086.format(name=name, listener=listener))
    return nlp