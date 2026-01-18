from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.utils import create_model
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.output_parsers.regex import RegexParser
@root_validator(pre=True)
def get_default_document_variable_name(cls, values: Dict) -> Dict:
    """Get default document variable name, if not provided."""
    if 'document_variable_name' not in values:
        llm_chain_variables = values['llm_chain'].prompt.input_variables
        if len(llm_chain_variables) == 1:
            values['document_variable_name'] = llm_chain_variables[0]
        else:
            raise ValueError('document_variable_name must be provided if there are multiple llm_chain input_variables')
    else:
        llm_chain_variables = values['llm_chain'].prompt.input_variables
        if values['document_variable_name'] not in llm_chain_variables:
            raise ValueError(f'document_variable_name {values['document_variable_name']} was not found in llm_chain input_variables: {llm_chain_variables}')
    return values