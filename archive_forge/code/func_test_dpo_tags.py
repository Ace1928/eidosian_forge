import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from .testing_utils import require_bitsandbytes, require_no_wandb, require_peft
@require_peft
def test_dpo_tags(self):
    model_id = 'HuggingFaceM4/tiny-random-LlamaForCausalLM'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=3, remove_unused_columns=False, gradient_accumulation_steps=4, learning_rate=0.9, evaluation_strategy='steps')
        dummy_dataset = self._init_dummy_dataset()
        trainer = DPOTrainer(model=model, ref_model=None, beta=0.1, args=training_args, tokenizer=tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset)
        assert trainer.model.model_tags == trainer._tag_names