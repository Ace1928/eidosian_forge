import unittest
import torch
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
def test_data_collator_finds_response_template_llama2_tokenizer(self):
    self.tokenizer = AutoTokenizer.from_pretrained('trl-internal-testing/dummy-GPT2-correct-vocab')
    self.instruction = '### System: You are a helpful assistant.\n\n### User: How much is 2+2?\n\n### Assistant: 2+2 equals 4'
    self.instruction_template = '\n### User:'
    self.response_template = '\n### Assistant:'
    self.tokenized_instruction_w_context = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)[1:]
    self.tokenized_response_w_context = self.tokenizer.encode(self.response_template, add_special_tokens=False)[2:]
    assert self.response_template in self.instruction
    self.tokenized_instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)
    self.collator = DataCollatorForCompletionOnlyLM(self.tokenized_response_w_context, tokenizer=self.tokenizer)
    self.collator.torch_call([self.tokenized_instruction])
    self.collator = DataCollatorForCompletionOnlyLM(self.tokenized_response_w_context, self.tokenized_instruction_w_context, tokenizer=self.tokenizer)
    self.collator.torch_call([self.tokenized_instruction])
    self.instruction = '## User: First instruction\n\n### Assistant: First response\n\n### User: Second instruction\n\n### Assistant: Second response'
    self.tokenized_instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)
    self.collator = DataCollatorForCompletionOnlyLM(self.tokenized_response_w_context, self.tokenized_instruction_w_context, tokenizer=self.tokenizer)
    collator_output = self.collator.torch_call([self.tokenized_instruction])
    collator_text = self.tokenizer.decode(collator_output['labels'][torch.where(collator_output['labels'] != -100)])
    expected_text = ' First response\n\n Second response'
    assert collator_text == expected_text