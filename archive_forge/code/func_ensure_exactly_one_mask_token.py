from typing import Dict
import numpy as np
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import GenericTensor, Pipeline, PipelineException, build_pipeline_init_args
def ensure_exactly_one_mask_token(self, model_inputs: GenericTensor):
    if isinstance(model_inputs, list):
        for model_input in model_inputs:
            self._ensure_exactly_one_mask_token(model_input['input_ids'][0])
    else:
        for input_ids in model_inputs['input_ids']:
            self._ensure_exactly_one_mask_token(input_ids)