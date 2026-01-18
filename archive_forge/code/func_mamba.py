from typing import TYPE_CHECKING, Optional
import torch
from .transformers import TransformerTokenizer
def mamba(model_name: str, device: Optional[str]=None, model_kwargs: dict={}, tokenizer_kwargs: dict={}):
    try:
        from mamba_ssm import MambaLMHeadModel
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError('The `mamba_ssm` library needs to be installed in order to use Mamba people.')
    if not torch.cuda.is_available():
        raise NotImplementedError('Mamba models can only run on GPU.')
    elif device is None:
        device = 'cuda'
    model = MambaLMHeadModel.from_pretrained(model_name, device=device)
    tokenizer_kwargs.setdefault('padding_side', 'left')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, **tokenizer_kwargs)
    return Mamba(model, tokenizer, device)