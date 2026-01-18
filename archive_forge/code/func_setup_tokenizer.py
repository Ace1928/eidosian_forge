import pytest
from transformers import AutoTokenizer
@pytest.fixture
def setup_tokenizer(llama_tokenizer, llama_version):

    def _helper(tokenizer_mock):
        tokenizer_mock.from_pretrained.return_value = llama_tokenizer[llama_version]
    return _helper