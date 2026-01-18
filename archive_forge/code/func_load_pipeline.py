from ._base import *
def load_pipeline(self, task_name='sentiment-analysis', tokenizer=None, device='auto'):
    from transformers import AutoTokenizer, pipeline
    device_id = device
    if device == 'auto':
        device_id = -1 if self.get_device().type == 'cpu' else 0
    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
    return pipeline(task=task_name, model=self.model, tokenizer=tokenizer, device=device_id)