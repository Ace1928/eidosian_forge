import copy
import functools
import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from datasets import Dataset, DatasetDict
from datasets import load_dataset as datasets_load_dataset
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor
from .. import logging
def create_dataset_processing_func(self, data_keys: Dict[str, str], ref_keys: Optional[List[str]]=None) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    return functools.partial(self.dataset_processing_func, data_keys=data_keys, ref_keys=ref_keys)