import unittest
from typing import Callable
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.models.utils import ChatMlSpecialTokens, setup_chat_format
def test_get_formatting_func_from_dataset_with_chatml_messages(self):
    dataset = Dataset.from_dict({'messages': [[{'role': 'system', 'content': 'You are helpful'}, {'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi, how can I help you?'}]]})
    formatting_func = get_formatting_func_from_dataset(dataset, self.llama_tokenizer)
    assert isinstance(formatting_func, Callable)
    formatted_text = formatting_func(dataset[0])
    expected = '<s>[INST] <<SYS>>\nYou are helpful\n<</SYS>>\n\nHello [/INST] Hi, how can I help you? </s>'
    assert formatted_text == expected
    formatted_text = formatting_func(dataset[0:1])
    assert formatted_text == [expected]
    formatting_func = get_formatting_func_from_dataset(dataset, self.chatml_tokenizer)
    formatted_text = formatting_func(dataset[0])
    expected = '<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi, how can I help you?<|im_end|>\n'
    assert formatted_text == expected
    formatted_text = formatting_func(dataset[0:1])
    assert formatted_text == [expected]