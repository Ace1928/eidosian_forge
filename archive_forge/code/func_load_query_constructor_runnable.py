from __future__ import annotations
import json
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.runnables import Runnable
from langchain.chains.llm import LLMChain
from langchain.chains.query_constructor.ir import (
from langchain.chains.query_constructor.parser import get_parser
from langchain.chains.query_constructor.prompt import (
from langchain.chains.query_constructor.schema import AttributeInfo
def load_query_constructor_runnable(llm: BaseLanguageModel, document_contents: str, attribute_info: Sequence[Union[AttributeInfo, dict]], *, examples: Optional[Sequence]=None, allowed_comparators: Sequence[Comparator]=tuple(Comparator), allowed_operators: Sequence[Operator]=tuple(Operator), enable_limit: bool=False, schema_prompt: Optional[BasePromptTemplate]=None, fix_invalid: bool=False, **kwargs: Any) -> Runnable:
    """Load a query constructor runnable chain.

    Args:
        llm: BaseLanguageModel to use for the chain.
        document_contents: Description of the page contents of the document to be
            queried.
        attribute_info: Sequence of attributes in the document.
        examples: Optional list of examples to use for the chain.
        allowed_comparators: Sequence of allowed comparators. Defaults to all
            Comparators.
        allowed_operators: Sequence of allowed operators. Defaults to all Operators.
        enable_limit: Whether to enable the limit operator. Defaults to False.
        schema_prompt: Prompt for describing query schema. Should have string input
            variables allowed_comparators and allowed_operators.
        fix_invalid: Whether to fix invalid filter directives by ignoring invalid
            operators, comparators and attributes.
        **kwargs: Additional named params to pass to FewShotPromptTemplate init.

    Returns:
        A Runnable that can be used to construct queries.
    """
    prompt = get_query_constructor_prompt(document_contents, attribute_info, examples=examples, allowed_comparators=allowed_comparators, allowed_operators=allowed_operators, enable_limit=enable_limit, schema_prompt=schema_prompt, **kwargs)
    allowed_attributes = []
    for ainfo in attribute_info:
        allowed_attributes.append(ainfo.name if isinstance(ainfo, AttributeInfo) else ainfo['name'])
    output_parser = StructuredQueryOutputParser.from_components(allowed_comparators=allowed_comparators, allowed_operators=allowed_operators, allowed_attributes=allowed_attributes, fix_invalid=fix_invalid)
    return prompt | llm | output_parser