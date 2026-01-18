import inspect
import torch
import torch.nn as nn
def is_adaption_prompt_trainable(params: str) -> bool:
    """Return True if module is trainable under adaption prompt fine-tuning."""
    return params.split('.')[-1].startswith('adaption_')