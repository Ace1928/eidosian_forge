from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
@classmethod
def from_llm_and_api_docs(cls, llm: BaseLanguageModel, api_docs: str, headers: Optional[dict]=None, api_url_prompt: BasePromptTemplate=API_URL_PROMPT, api_response_prompt: BasePromptTemplate=API_RESPONSE_PROMPT, limit_to_domains: Optional[Sequence[str]]=tuple(), **kwargs: Any) -> APIChain:
    """Load chain from just an LLM and the api docs."""
    get_request_chain = LLMChain(llm=llm, prompt=api_url_prompt)
    requests_wrapper = TextRequestsWrapper(headers=headers)
    get_answer_chain = LLMChain(llm=llm, prompt=api_response_prompt)
    return cls(api_request_chain=get_request_chain, api_answer_chain=get_answer_chain, requests_wrapper=requests_wrapper, api_docs=api_docs, limit_to_domains=limit_to_domains, **kwargs)