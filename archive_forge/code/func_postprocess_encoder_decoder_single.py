import re
from typing import List, Optional, Tuple, Union
import numpy as np
from ..utils import (
from .base import ChunkPipeline, build_pipeline_init_args
from .question_answering import select_starts_ends
def postprocess_encoder_decoder_single(self, model_outputs, **kwargs):
    sequence = self.tokenizer.batch_decode(model_outputs['sequences'])[0]
    sequence = sequence.replace(self.tokenizer.eos_token, '').replace(self.tokenizer.pad_token, '')
    sequence = re.sub('<.*?>', '', sequence, count=1).strip()
    ret = {'answer': None}
    answer = re.search('<s_answer>(.*)</s_answer>', sequence)
    if answer is not None:
        ret['answer'] = answer.group(1).strip()
    return ret