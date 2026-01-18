import json
import re
from typing import Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.api.openapi.prompts import REQUEST_TEMPLATE
from langchain.chains.llm import LLMChain
@classmethod
def from_llm_and_typescript(cls, llm: BaseLanguageModel, typescript_definition: str, verbose: bool=True, **kwargs: Any) -> LLMChain:
    """Get the request parser."""
    output_parser = APIRequesterOutputParser()
    prompt = PromptTemplate(template=REQUEST_TEMPLATE, output_parser=output_parser, partial_variables={'schema': typescript_definition}, input_variables=['instructions'])
    return cls(prompt=prompt, llm=llm, verbose=verbose, **kwargs)