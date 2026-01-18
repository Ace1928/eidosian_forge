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
def test_data_collator_completion_lm(self):
    response_template = '### Response:\n'
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer, mlm=False)
    text = '\n\n### Instructions:\nHello all this should be masked\n\n### Response:\nI have not been masked correctly.'
    encoded_text = self.tokenizer(text)
    examples = [encoded_text]
    batch = data_collator(examples)
    labels = batch['labels']
    last_pad_idx = np.where(labels == -100)[1][-1]
    result_text = self.tokenizer.decode(batch['input_ids'][0, last_pad_idx + 1:])
    assert result_text == 'I have not been masked correctly.'