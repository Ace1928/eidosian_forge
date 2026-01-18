from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
def setup_chat_format(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, format: Optional[Literal['chatml']]='chatml', resize_to_multiple_of: Optional[int]=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format, and extending the embedding layer of the model based on the new special tokens.

    Args:
      model (`~transformers.PreTrainedModel`): The model to be modified.
      tokenizer (`~transformers.PreTrainedTokenizer`): The tokenizer to be modified.
      format (`Optional[Literal["chatml"]]`): The format to be set. Defaults to "chatml".
      resize_to_multiple_of (`Optional[int]`): Number to resize the embedding layer to. Defaults to None.
    Returns:
      model (`~transformers.PreTrainedModel`): The modified model.
      tokenizer (`~transformers.PreTrainedTokenizer`): The modified tokenizer.
    """
    if format not in FORMAT_MAPPING:
        raise ValueError(f'Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}')
    chat_format = FORMAT_MAPPING[format]()
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens({'additional_special_tokens': [chat_format.bos_token, chat_format.eos_token]})
    tokenizer.chat_template = chat_format.chat_template
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None)
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return (model, tokenizer)