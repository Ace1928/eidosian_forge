import os
import tempfile
import unittest
import torch
from pytest import mark
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, is_peft_available
from .testing_utils import require_bitsandbytes, require_peft
def test_check_peft_model_nb_trainable_params(self):
    """
        Check that the number of trainable parameters is correct.
        """
    causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
    pretrained_model = get_peft_model(causal_lm_model, self.lora_config)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
    nb_trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    assert nb_trainable_params == 10273
    non_peft_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id)
    nb_trainable_params = sum((p.numel() for p in non_peft_model.parameters() if p.requires_grad))
    assert nb_trainable_params == 99578