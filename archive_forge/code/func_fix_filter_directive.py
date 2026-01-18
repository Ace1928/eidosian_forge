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
def fix_filter_directive(filter: Optional[FilterDirective], *, allowed_comparators: Optional[Sequence[Comparator]]=None, allowed_operators: Optional[Sequence[Operator]]=None, allowed_attributes: Optional[Sequence[str]]=None) -> Optional[FilterDirective]:
    """Fix invalid filter directive.

    Args:
        filter: Filter directive to fix.
        allowed_comparators: allowed comparators. Defaults to all comparators.
        allowed_operators: allowed operators. Defaults to all operators.
        allowed_attributes: allowed attributes. Defaults to all attributes.

    Returns:
        Fixed filter directive.
    """
    if not (allowed_comparators or allowed_operators or allowed_attributes) or not filter:
        return filter
    elif isinstance(filter, Comparison):
        if allowed_comparators and filter.comparator not in allowed_comparators:
            return None
        if allowed_attributes and filter.attribute not in allowed_attributes:
            return None
        return filter
    elif isinstance(filter, Operation):
        if allowed_operators and filter.operator not in allowed_operators:
            return None
        args = [cast(FilterDirective, fix_filter_directive(arg, allowed_comparators=allowed_comparators, allowed_operators=allowed_operators, allowed_attributes=allowed_attributes)) for arg in filter.arguments if arg is not None]
        if not args:
            return None
        elif len(args) == 1 and filter.operator in (Operator.AND, Operator.OR):
            return args[0]
        else:
            return Operation(operator=filter.operator, arguments=args)
    else:
        return filter