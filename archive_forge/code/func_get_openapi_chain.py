from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities.openapi import OpenAPISpec
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.utils.input import get_colored_text
from requests import Response
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.tools import APIOperation
def get_openapi_chain(spec: Union[OpenAPISpec, str], llm: Optional[BaseLanguageModel]=None, prompt: Optional[BasePromptTemplate]=None, request_chain: Optional[Chain]=None, llm_chain_kwargs: Optional[Dict]=None, verbose: bool=False, headers: Optional[Dict]=None, params: Optional[Dict]=None, **kwargs: Any) -> SequentialChain:
    """Create a chain for querying an API from a OpenAPI spec.

    Args:
        spec: OpenAPISpec or url/file/text string corresponding to one.
        llm: language model, should be an OpenAI function-calling model, e.g.
            `ChatOpenAI(model="gpt-3.5-turbo-0613")`.
        prompt: Main prompt template to use.
        request_chain: Chain for taking the functions output and executing the request.
    """
    if isinstance(spec, str):
        for conversion in (OpenAPISpec.from_url, OpenAPISpec.from_file, OpenAPISpec.from_text):
            try:
                spec = conversion(spec)
                break
            except ImportError as e:
                raise e
            except Exception:
                pass
        if isinstance(spec, str):
            raise ValueError(f'Unable to parse spec from source {spec}')
    openai_fns, call_api_fn = openapi_spec_to_openai_fn(spec)
    llm = llm or ChatOpenAI(model='gpt-3.5-turbo-0613')
    prompt = prompt or ChatPromptTemplate.from_template("Use the provided API's to respond to this user query:\n\n{query}")
    llm_chain = LLMChain(llm=llm, prompt=prompt, llm_kwargs={'functions': openai_fns}, output_parser=JsonOutputFunctionsParser(args_only=False), output_key='function', verbose=verbose, **llm_chain_kwargs or {})
    request_chain = request_chain or SimpleRequestChain(request_method=lambda name, args: call_api_fn(name, args, headers=headers, params=params), verbose=verbose)
    return SequentialChain(chains=[llm_chain, request_chain], input_variables=llm_chain.input_keys, output_variables=['response'], verbose=verbose, **kwargs)