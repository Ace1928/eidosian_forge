import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_from_save_transformers(self):
    """
        Test if the model can be saved and loaded using transformers and get the same weights.
        We override the test of the super class to check if the weights are the same.
        """
    for model_name in self.all_model_names:
        transformers_model = self.trl_model_class.transformers_parent_class.from_pretrained(model_name)
        trl_model = self.trl_model_class.from_pretrained(model_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            trl_model.save_pretrained(tmp_dir)
            transformers_model_from_save = self.trl_model_class.transformers_parent_class.from_pretrained(tmp_dir)
        for key in transformers_model.state_dict():
            assert torch.allclose(transformers_model_from_save.state_dict()[key], transformers_model.state_dict()[key])
        for key in trl_model.state_dict():
            if 'v_head' not in key:
                assert key in transformers_model.state_dict()
                assert torch.allclose(trl_model.state_dict()[key], transformers_model.state_dict()[key])
        assert set(transformers_model_from_save.state_dict().keys()) == set(transformers_model.state_dict().keys())