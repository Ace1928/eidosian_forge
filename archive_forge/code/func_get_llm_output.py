from typing import (
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_community.chat_models.litellm import (
def get_llm_output(usage: Any, **params: Any) -> dict:
    """Get llm output from usage and params."""
    llm_output = {token_usage_key_name: usage}
    metadata = params['metadata']
    for key in metadata:
        if key not in llm_output:
            llm_output[key] = metadata[key]
    return llm_output