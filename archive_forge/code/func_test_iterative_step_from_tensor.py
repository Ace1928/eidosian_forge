import tempfile
import unittest
import torch
from datasets import Dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import IterativeSFTTrainer
@parameterized.expand([['gpt2', 'tensor'], ['gpt2', 'text'], ['t5', 'tensor'], ['t5', 'text']])
def test_iterative_step_from_tensor(self, model_name, input_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        if input_name == 'tensor':
            dummy_dataset = self._init_tensor_dummy_dataset()
            inputs = {'input_ids': dummy_dataset['input_ids'], 'attention_mask': dummy_dataset['attention_mask'], 'labels': dummy_dataset['labels']}
        else:
            dummy_dataset = self._init_textual_dummy_dataset()
            inputs = {'texts': dummy_dataset['texts'], 'texts_labels': dummy_dataset['texts_labels']}
        if model_name == 'gpt2':
            model = self.model
            tokenizer = self.tokenizer
        else:
            model = self.t5_model
            tokenizer = self.t5_tokenizer
        args = TrainingArguments(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=2)
        iterative_trainer = IterativeSFTTrainer(model=model, args=args, tokenizer=tokenizer)
        iterative_trainer.step(**inputs)
        for param in iterative_trainer.model.parameters():
            assert param.grad is not None