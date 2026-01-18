import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from .testing_utils import require_bitsandbytes, require_no_wandb, require_peft
def test_dpo_trainer_without_providing_ref_model(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=3, remove_unused_columns=False, gradient_accumulation_steps=4, learning_rate=0.9, evaluation_strategy='steps')
        dummy_dataset = self._init_dummy_dataset()
        trainer = DPOTrainer(model=self.model, ref_model=None, beta=0.1, args=training_args, tokenizer=self.tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset, precompute_ref_log_probs=True)
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        trainer.train()
        assert trainer.state.log_history[-1]['train_loss'] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:
                assert not torch.equal(param, new_param)