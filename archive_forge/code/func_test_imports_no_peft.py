import sys
import unittest
from unittest.mock import patch
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .testing_utils import is_peft_available, require_peft
def test_imports_no_peft(self):
    with patch.dict(sys.modules, {'peft': None}):
        from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, PreTrainedModelWrapper