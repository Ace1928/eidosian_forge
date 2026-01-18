import gc
import itertools
import tempfile
import unittest
import torch
from accelerate.utils.memory import release_memory
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer, is_peft_available
from ..testing_utils import require_bitsandbytes, require_peft, require_torch_gpu
from .testing_constants import DPO_LOSS_TYPES, DPO_PRECOMPUTE_LOGITS, GRADIENT_CHECKPOINTING_KWARGS, MODELS_TO_TEST
@parameterized.expand(list(itertools.product(MODELS_TO_TEST, DPO_LOSS_TYPES, DPO_PRECOMPUTE_LOGITS, GRADIENT_CHECKPOINTING_KWARGS)))
@require_bitsandbytes
@require_peft
def test_dpo_peft_model_qlora(self, model_id, loss_type, pre_compute_logits, gradient_checkpointing_kwargs):
    """
        A test that tests the simple usage of `DPOTrainer` using QLoRA + different scenarios of gradient checkpointing.
        """
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=2, remove_unused_columns=False, gradient_accumulation_steps=2, learning_rate=0.9, evaluation_strategy='steps', fp16=True, logging_strategy='no', report_to='none', gradient_checkpointing=True, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        trainer = DPOTrainer(model=model, ref_model=None, beta=0.1, args=training_args, tokenizer=tokenizer, train_dataset=self.dataset, eval_dataset=self.dataset, generate_during_eval=False, loss_type=loss_type, precompute_ref_log_probs=pre_compute_logits, peft_config=self.peft_config, max_length=self.max_length)
        assert isinstance(trainer.model, PeftModel)
        assert trainer.ref_model is None
        trainer.train()
        trainer.save_model()
    release_memory(model, trainer)