import unittest
import torch
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
def test_data_collator_handling_of_long_sequences(self):
    self.tokenizer = AutoTokenizer.from_pretrained('trl-internal-testing/dummy-GPT2-correct-vocab')
    self.instruction = "### System: You are a helpful assistant.\n\n### User: How much is 2+2? I'm asking because I'm not sure. And I'm not sure because I'm not good at math.\n"
    self.response_template = '\n### Assistant:'
    self.tokenized_instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)
    self.collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer=self.tokenizer)
    encoded_instance = self.collator.torch_call([self.tokenized_instruction])
    result = torch.all(encoded_instance['labels'] == -100)
    assert result, 'Not all values in the tensor are -100.'
    self.instruction_template = '\n### User:'
    self.collator = DataCollatorForCompletionOnlyLM(self.response_template, self.instruction_template, tokenizer=self.tokenizer)
    encoded_instance = self.collator.torch_call([self.tokenized_instruction])
    result = torch.all(encoded_instance['labels'] == -100)
    assert result, 'Not all values in the tensor are -100.'