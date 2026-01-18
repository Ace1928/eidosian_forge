import copy
import os
import tempfile
import unittest
import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from trl.import_utils import is_peft_available
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM
from .testing_utils import require_peft
def test_data_collator_chat_completion_lm(self):
    instruction_template = '### Human:'
    assistant_template = '### Assistant:'
    data_collator = DataCollatorForCompletionOnlyLM(response_template=assistant_template, instruction_template=instruction_template, tokenizer=self.tokenizer, mlm=False)
    text = '### Human: Hello all this should be masked.### Assistant: I should not be masked.### Human: All this should be masked too.### Assistant: I should not be masked too.'
    encoded_text = self.tokenizer(text)
    examples = [encoded_text]
    batch = data_collator(examples)
    labels = batch['labels']
    non_masked_tokens = batch['input_ids'][labels != -100]
    result_text = self.tokenizer.decode(non_masked_tokens)
    assert result_text == ' I should not be masked. I should not be masked too.'