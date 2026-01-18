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
def construct_examples(input_output_pairs: Sequence[Tuple[str, dict]]) -> List[dict]:
    """Construct examples from input-output pairs.

    Args:
        input_output_pairs: Sequence of input-output pairs.

    Returns:
        List of examples.
    """
    examples = []
    for i, (_input, output) in enumerate(input_output_pairs):
        structured_request = json.dumps(output, indent=4).replace('{', '{{').replace('}', '}}')
        example = {'i': i + 1, 'user_query': _input, 'structured_request': structured_request}
        examples.append(example)
    return examples