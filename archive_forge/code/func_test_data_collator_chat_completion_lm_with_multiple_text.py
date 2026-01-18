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
def test_data_collator_chat_completion_lm_with_multiple_text(self):
    tokenizer = copy.deepcopy(self.tokenizer)
    tokenizer.padding_side = 'left'
    instruction_template = '### Human:'
    assistant_template = '### Assistant:'
    data_collator = DataCollatorForCompletionOnlyLM(response_template=assistant_template, instruction_template=instruction_template, tokenizer=tokenizer, mlm=False)
    text1 = '### Human: Hello all this should be masked.### Assistant: I should not be masked.'
    text2 = '### Human: Hello all this should be masked.### Assistant: I should not be masked.### Human: All this should be masked too.### Assistant: I should not be masked too.'
    encoded_text1 = tokenizer(text1)
    encoded_text2 = tokenizer(text2)
    examples = [encoded_text1, encoded_text2]
    batch = data_collator(examples)
    labels = batch['labels']
    input_ids = batch['input_ids']
    non_masked_tokens1 = input_ids[0][labels[0] != -100]
    result_text1 = tokenizer.decode(non_masked_tokens1)
    assert result_text1 == ' I should not be masked.'
    non_masked_tokens2 = input_ids[1][labels[1] != -100]
    result_text2 = tokenizer.decode(non_masked_tokens2)
    assert result_text2 == ' I should not be masked. I should not be masked too.'