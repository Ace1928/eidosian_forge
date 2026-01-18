from typing import Dict
import numpy as np
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import GenericTensor, Pipeline, PipelineException, build_pipeline_init_args
def get_target_ids(self, targets, top_k=None):
    if isinstance(targets, str):
        targets = [targets]
    try:
        vocab = self.tokenizer.get_vocab()
    except Exception:
        vocab = {}
    target_ids = []
    for target in targets:
        id_ = vocab.get(target, None)
        if id_ is None:
            input_ids = self.tokenizer(target, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, max_length=1, truncation=True)['input_ids']
            if len(input_ids) == 0:
                logger.warning(f'The specified target token `{target}` does not exist in the model vocabulary. We cannot replace it with anything meaningful, ignoring it')
                continue
            id_ = input_ids[0]
            logger.warning(f'The specified target token `{target}` does not exist in the model vocabulary. Replacing with `{self.tokenizer.convert_ids_to_tokens(id_)}`.')
        target_ids.append(id_)
    target_ids = list(set(target_ids))
    if len(target_ids) == 0:
        raise ValueError('At least one target must be provided when passed.')
    target_ids = np.array(target_ids)
    return target_ids