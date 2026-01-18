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
@require_bitsandbytes
@mark.peft_test
def test_dpo_lora_bf16_autocast_llama(self):
    from peft import LoraConfig
    model_id = 'HuggingFaceM4/tiny-random-LlamaForCausalLM'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=3, remove_unused_columns=False, gradient_accumulation_steps=4, learning_rate=0.9, evaluation_strategy='steps', bf16=True)
        dummy_dataset = self._init_dummy_dataset()
        trainer = DPOTrainer(model=model, ref_model=None, beta=0.1, args=training_args, tokenizer=tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset, peft_config=lora_config, generate_during_eval=True)
        trainer.train()
        trainer.save_model()