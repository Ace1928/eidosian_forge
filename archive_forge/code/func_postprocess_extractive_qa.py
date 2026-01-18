import re
from typing import List, Optional, Tuple, Union
import numpy as np
from ..utils import (
from .base import ChunkPipeline, build_pipeline_init_args
from .question_answering import select_starts_ends
def postprocess_extractive_qa(self, model_outputs, top_k=1, handle_impossible_answer=False, max_answer_len=15, **kwargs):
    min_null_score = 1000000
    answers = []
    for output in model_outputs:
        words = output['words']
        starts, ends, scores, min_null_score = select_starts_ends(start=output['start_logits'], end=output['end_logits'], p_mask=output['p_mask'], attention_mask=output['attention_mask'].numpy() if output.get('attention_mask', None) is not None else None, min_null_score=min_null_score, top_k=top_k, handle_impossible_answer=handle_impossible_answer, max_answer_len=max_answer_len)
        word_ids = output['word_ids']
        for start, end, score in zip(starts, ends, scores):
            word_start, word_end = (word_ids[start], word_ids[end])
            if word_start is not None and word_end is not None:
                answers.append({'score': float(score), 'answer': ' '.join(words[word_start:word_end + 1]), 'start': word_start, 'end': word_end})
    if handle_impossible_answer:
        answers.append({'score': min_null_score, 'answer': '', 'start': 0, 'end': 0})
    return answers