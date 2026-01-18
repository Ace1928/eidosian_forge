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
def ast_parse(raw_filter: str) -> Optional[FilterDirective]:
    filter = cast(Optional[FilterDirective], get_parser().parse(raw_filter))
    fixed = fix_filter_directive(filter, allowed_comparators=allowed_comparators, allowed_operators=allowed_operators, allowed_attributes=allowed_attributes)
    return fixed