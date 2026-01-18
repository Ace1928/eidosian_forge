import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_from_save_transformers_sharded(self):
    """
        Test if the model can be saved and loaded using transformers and get the same weights - sharded case
        """
    for model_name in self.all_model_names:
        transformers_model = self.trl_model_class.transformers_parent_class.from_pretrained(model_name)
        trl_model = self.trl_model_class.from_pretrained(model_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            trl_model.save_pretrained(tmp_dir, max_shard_size='1MB')
            transformers_model_from_save = self.trl_model_class.transformers_parent_class.from_pretrained(tmp_dir)
        for key in transformers_model.state_dict():
            assert torch.allclose(transformers_model_from_save.state_dict()[key], transformers_model.state_dict()[key])