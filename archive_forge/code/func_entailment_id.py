import inspect
from typing import List, Union
import numpy as np
from ..tokenization_utils import TruncationStrategy
from ..utils import add_end_docstrings, logging
from .base import ArgumentHandler, ChunkPipeline, build_pipeline_init_args
@property
def entailment_id(self):
    for label, ind in self.model.config.label2id.items():
        if label.lower().startswith('entail'):
            return ind
    return -1