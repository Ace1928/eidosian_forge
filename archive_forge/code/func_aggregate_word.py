import types
import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
from ..models.bert.tokenization_bert import BasicTokenizer
from ..utils import (
from .base import ArgumentHandler, ChunkPipeline, Dataset, build_pipeline_init_args
def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
    word = self.tokenizer.convert_tokens_to_string([entity['word'] for entity in entities])
    if aggregation_strategy == AggregationStrategy.FIRST:
        scores = entities[0]['scores']
        idx = scores.argmax()
        score = scores[idx]
        entity = self.model.config.id2label[idx]
    elif aggregation_strategy == AggregationStrategy.MAX:
        max_entity = max(entities, key=lambda entity: entity['scores'].max())
        scores = max_entity['scores']
        idx = scores.argmax()
        score = scores[idx]
        entity = self.model.config.id2label[idx]
    elif aggregation_strategy == AggregationStrategy.AVERAGE:
        scores = np.stack([entity['scores'] for entity in entities])
        average_scores = np.nanmean(scores, axis=0)
        entity_idx = average_scores.argmax()
        entity = self.model.config.id2label[entity_idx]
        score = average_scores[entity_idx]
    else:
        raise ValueError('Invalid aggregation_strategy')
    new_entity = {'entity': entity, 'score': score, 'word': word, 'start': entities[0]['start'], 'end': entities[-1]['end']}
    return new_entity