import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from .base import TaskProcessor
def create_defaults_and_kwargs_from_preprocessor_kwargs(self, preprocessor_kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if preprocessor_kwargs is None:
        preprocessor_kwargs = {}
    kwargs = copy.deepcopy(preprocessor_kwargs)
    defaults = {}
    defaults['padding'] = kwargs.pop('padding', 'max_length')
    defaults['truncation'] = kwargs.pop('truncation', True)
    defaults['max_length'] = kwargs.pop('max_length', self.preprocessor.model_max_length)
    defaults['is_split_into_words'] = kwargs.pop('is_split_into_words', True)
    return (defaults, kwargs)