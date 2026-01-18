import types
import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
from ..models.bert.tokenization_bert import BasicTokenizer
from ..utils import (
from .base import ArgumentHandler, ChunkPipeline, Dataset, build_pipeline_init_args
def group_sub_entities(self, entities: List[dict]) -> dict:
    """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
    entity = entities[0]['entity'].split('-', 1)[-1]
    scores = np.nanmean([entity['score'] for entity in entities])
    tokens = [entity['word'] for entity in entities]
    entity_group = {'entity_group': entity, 'score': np.mean(scores), 'word': self.tokenizer.convert_tokens_to_string(tokens), 'start': entities[0]['start'], 'end': entities[-1]['end']}
    return entity_group