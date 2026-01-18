import unittest
from typing import Callable
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.models.utils import ChatMlSpecialTokens, setup_chat_format
def test_get_formatting_func_from_dataset_from_hub(self):
    ds_1 = load_dataset('philschmid/trl-test-instruction', split='train')
    ds_2 = load_dataset('philschmid/dolly-15k-oai-style', split='train')
    for ds in [ds_1, ds_2]:
        formatting_func = get_formatting_func_from_dataset(ds, self.llama_tokenizer)
        assert formatting_func is not None
        assert isinstance(formatting_func, Callable)
    ds_3 = load_dataset('philschmid/guanaco-sharegpt-style', split='train')
    formatting_func = get_formatting_func_from_dataset(ds_3, self.llama_tokenizer)
    assert formatting_func is None