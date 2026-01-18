import logging
from typing import Callable, Literal, Optional, Union
from datasets import Dataset, Value
from transformers import AutoTokenizer
from ..trainer.utils import ConstantLengthDataset
def get_formatting_func_from_dataset(dataset: Union[Dataset, ConstantLengthDataset], tokenizer: AutoTokenizer) -> Optional[Callable]:
    """
    Finds the correct formatting function based on the dataset structure. Currently supported datasets are:
    - `ChatML` with [{"role": str, "content": str}]
    - `instruction` with [{"prompt": str, "completion": str}]

    Args:
        dataset (Dataset): User dataset
        tokenizer (AutoTokenizer): Tokenizer used for formatting

    Returns:
        Callable: Formatting function if the dataset format is supported else None
    """
    if isinstance(dataset, Dataset):
        if 'messages' in dataset.features:
            if dataset.features['messages'] == FORMAT_MAPPING['chatml']:
                logging.info('Formatting dataset with chatml format')
                return conversations_formatting_function(tokenizer, 'messages')
        if 'conversations' in dataset.features:
            if dataset.features['conversations'] == FORMAT_MAPPING['chatml']:
                logging.info('Formatting dataset with chatml format')
                return conversations_formatting_function(tokenizer, 'conversations')
        elif dataset.features == FORMAT_MAPPING['instruction']:
            logging.info('Formatting dataset with instruction format')
            return instructions_formatting_function(tokenizer)
    return None