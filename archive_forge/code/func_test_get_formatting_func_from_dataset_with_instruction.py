import unittest
from typing import Callable
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.models.utils import ChatMlSpecialTokens, setup_chat_format
def test_get_formatting_func_from_dataset_with_instruction(self):
    dataset = Dataset.from_list([{'prompt': 'What is 2+2?', 'completion': '4'}, {'prompt': 'What is 3+3?', 'completion': '6'}])
    formatting_func = get_formatting_func_from_dataset(dataset, self.llama_tokenizer)
    assert formatting_func is not None
    assert isinstance(formatting_func, Callable)
    formatted_text = formatting_func(dataset[0])
    assert formatted_text == '<s>[INST] What is 2+2? [/INST] 4 </s>'
    formatted_text = formatting_func(dataset[0:1])
    assert formatted_text == ['<s>[INST] What is 2+2? [/INST] 4 </s>']