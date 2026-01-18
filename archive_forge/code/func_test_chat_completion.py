import sys
from pathlib import Path
from typing import List, Literal, TypedDict
from unittest.mock import patch
import pytest
import torch
from llama_recipes.inference.chat_utils import read_dialogs_from_file
@pytest.mark.skip_missing_tokenizer
@patch('chat_completion.AutoTokenizer')
@patch('chat_completion.load_model')
def test_chat_completion(load_model, tokenizer, setup_tokenizer, llama_tokenizer, llama_version):
    from chat_completion import main
    setup_tokenizer(tokenizer)
    kwargs = {'prompt_file': (CHAT_COMPLETION_DIR / 'chats.json').as_posix()}
    main(llama_version, **kwargs)
    dialogs = read_dialogs_from_file(kwargs['prompt_file'])
    format_tokens = _format_tokens_llama2 if llama_version == 'meta-llama/Llama-2-7b-hf' else _format_tokens_llama3
    REF_RESULT = format_tokens(dialogs, llama_tokenizer[llama_version])
    assert all((load_model.return_value.generate.mock_calls[0 * 4][2]['input_ids'].cpu() == torch.tensor(REF_RESULT[0]).long()).tolist())
    assert all((load_model.return_value.generate.mock_calls[1 * 4][2]['input_ids'].cpu() == torch.tensor(REF_RESULT[1]).long()).tolist())
    assert all((load_model.return_value.generate.mock_calls[2 * 4][2]['input_ids'].cpu() == torch.tensor(REF_RESULT[2]).long()).tolist())
    assert all((load_model.return_value.generate.mock_calls[3 * 4][2]['input_ids'].cpu() == torch.tensor(REF_RESULT[3]).long()).tolist())
    assert all((load_model.return_value.generate.mock_calls[4 * 4][2]['input_ids'].cpu() == torch.tensor(REF_RESULT[4]).long()).tolist())