import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from .base import TaskProcessor
def dataset_processing_func(self, example: Dict[str, Any], data_keys: Dict[str, str], ref_keys: Optional[List[str]]=None) -> Dict[str, Any]:
    tokenized_inputs = self.preprocessor(text=example[data_keys['primary']], **self.defaults, **self.preprocessor_kwargs)
    return tokenized_inputs