import dataclasses
from typing import AbstractSet
from typing import Collection
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from .expression import Expression
from .expression import ParseError
from .structures import EMPTY_PARAMETERSET_OPTION
from .structures import get_empty_parameterset_mark
from .structures import Mark
from .structures import MARK_GEN
from .structures import MarkDecorator
from .structures import MarkGenerator
from .structures import ParameterSet
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.stash import StashKey
def deselect_by_keyword(items: 'List[Item]', config: Config) -> None:
    keywordexpr = config.option.keyword.lstrip()
    if not keywordexpr:
        return
    expr = _parse_expression(keywordexpr, "Wrong expression passed to '-k'")
    remaining = []
    deselected = []
    for colitem in items:
        if not expr.evaluate(KeywordMatcher.from_item(colitem)):
            deselected.append(colitem)
        else:
            remaining.append(colitem)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining