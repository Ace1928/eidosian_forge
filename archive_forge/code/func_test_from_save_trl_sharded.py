import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_from_save_trl_sharded(self):
    """
        Test if the model can be saved and loaded from a directory and get the same weights - sharded case
        """
    for model_name in self.all_model_names:
        model = self.trl_model_class.from_pretrained(model_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model_from_save = self.trl_model_class.from_pretrained(tmp_dir)
        for key in model_from_save.state_dict():
            assert torch.allclose(model_from_save.state_dict()[key], model.state_dict()[key])