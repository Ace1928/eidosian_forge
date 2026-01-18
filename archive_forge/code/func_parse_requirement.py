import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def parse_requirement(source: str) -> ParsedRequirement:
    return _parse_requirement(Tokenizer(source, rules=DEFAULT_RULES))