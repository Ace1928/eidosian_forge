import sys
import unittest
from unittest.mock import patch
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .testing_utils import is_peft_available, require_peft
def test_no_peft(self):
    with patch.dict(sys.modules, {'peft': None}):
        from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
        with pytest.raises(ModuleNotFoundError):
            import peft
        _trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id)
        _trl_seq2seq_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(self.seq_to_seq_model_id)