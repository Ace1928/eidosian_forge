from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import torch
from outlines.models.tokenizer import Tokenizer
def get_llama_tokenizer_types():
    """Get all the Llama tokenizer types/classes that need work-arounds.

    When they can't be imported, a dummy class is created.

    """
    try:
        from transformers.models.llama import LlamaTokenizer
    except ImportError:

        class LlamaTokenizer:
            pass
    try:
        from transformers.models.llama import LlamaTokenizerFast
    except ImportError:

        class LlamaTokenizerFast:
            pass
    try:
        from transformers.models.code_llama import CodeLlamaTokenizer
    except ImportError:

        class CodeLlamaTokenizer:
            pass
    try:
        from transformers.models.code_llama import CodeLlamaTokenizerFast
    except ImportError:

        class CodeLlamaTokenizerFast:
            pass
    return (LlamaTokenizer, LlamaTokenizerFast, CodeLlamaTokenizer, CodeLlamaTokenizerFast)