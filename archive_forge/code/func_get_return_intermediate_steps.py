from __future__ import annotations
from typing import Any, Dict, List, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain.chains.combine_documents.base import (
from langchain.chains.llm import LLMChain
@root_validator(pre=True)
def get_return_intermediate_steps(cls, values: Dict) -> Dict:
    """For backwards compatibility."""
    if 'return_refine_steps' in values:
        values['return_intermediate_steps'] = values['return_refine_steps']
        del values['return_refine_steps']
    return values