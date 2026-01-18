import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from .testing_utils import require_bitsandbytes, require_no_wandb, require_peft
@parameterized.expand([['gpt2', 'sigmoid', True], ['t5', 'hinge', False], ['gpt2', 'ipo', False], ['t5', 'ipo', True], ['gpt2', 'kto_pair', True], ['t5', 'kto_pair', False]])
def test_dpo_trainer(self, name, loss_type, pre_compute):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=3, remove_unused_columns=False, gradient_accumulation_steps=1, learning_rate=0.9, evaluation_strategy='steps')
        dummy_dataset = self._init_dummy_dataset()
        if name == 'gpt2':
            model = self.model
            ref_model = self.ref_model
            tokenizer = self.tokenizer
        elif name == 't5':
            model = self.t5_model
            ref_model = self.t5_ref_model
            tokenizer = self.t5_tokenizer
        trainer = DPOTrainer(model=model, ref_model=ref_model, beta=0.1, loss_type=loss_type, args=training_args, tokenizer=tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset, precompute_ref_log_probs=pre_compute)
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        trainer.train()
        assert trainer.state.log_history[-1]['train_loss'] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:
                assert not torch.equal(param, new_param)