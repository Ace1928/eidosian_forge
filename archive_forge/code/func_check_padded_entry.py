import pytest
from unittest.mock import patch
from transformers import LlamaTokenizer
def check_padded_entry(batch, tokenizer):
    seq_len = sum(batch['attention_mask'][0])
    assert seq_len < len(batch['attention_mask'][0])
    if tokenizer.vocab_size >= 128000:
        END_OF_TEXT_ID = 128009
    else:
        END_OF_TEXT_ID = tokenizer.eos_token_id
    assert batch['labels'][0][0] == -100
    assert batch['labels'][0][seq_len - 1] == END_OF_TEXT_ID
    assert batch['labels'][0][-1] == -100
    assert batch['input_ids'][0][0] == tokenizer.bos_token_id
    assert batch['input_ids'][0][-1] == tokenizer.eos_token_id