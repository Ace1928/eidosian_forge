import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from .testing_utils import require_bitsandbytes, require_no_wandb, require_peft
@require_no_wandb
def test_dpo_trainer_generate_during_eval_no_wandb(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=3, remove_unused_columns=False, gradient_accumulation_steps=1, learning_rate=0.9, evaluation_strategy='steps')
        dummy_dataset = self._init_dummy_dataset()
        with self.assertRaisesRegex(ValueError, expected_regex='`generate_during_eval=True` requires Weights and Biases to be installed. Please install `wandb` to resolve.'):
            DPOTrainer(model=self.model, ref_model=None, beta=0.1, args=training_args, tokenizer=self.tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset, generate_during_eval=True)