from __future__ import annotations
import asyncio
import functools
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.openai_tools import (
from langchain_core.outputs import (
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
from langchain_community.llms.tongyi import (
def stream_completion_with_retry(self, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(self)

    @retry_decorator
    def _stream_completion_with_retry(**_kwargs: Any) -> Any:
        responses = self.client.call(**_kwargs)
        for resp in responses:
            yield check_response(resp)
    return _stream_completion_with_retry(**kwargs)