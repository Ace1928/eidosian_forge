import collections.abc
import os
import pprint
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Protocol
from typing import Sequence
from unicodedata import normalize
from _pytest import outcomes
import _pytest._code
from _pytest._io.pprint import PrettyPrinter
from _pytest._io.saferepr import saferepr
from _pytest._io.saferepr import saferepr_unlimited
from _pytest.config import Config
def assertrepr_compare(config, op: str, left: Any, right: Any, use_ascii: bool=False) -> Optional[List[str]]:
    """Return specialised explanations for some operators/operands."""
    verbose = config.get_verbosity(Config.VERBOSITY_ASSERTIONS)
    use_ascii = isinstance(left, str) and isinstance(right, str) and (normalize('NFD', left) == normalize('NFD', right))
    if verbose > 1:
        left_repr = saferepr_unlimited(left, use_ascii=use_ascii)
        right_repr = saferepr_unlimited(right, use_ascii=use_ascii)
    else:
        maxsize = (80 - 15 - len(op) - 2) // 2
        left_repr = saferepr(left, maxsize=maxsize, use_ascii=use_ascii)
        right_repr = saferepr(right, maxsize=maxsize, use_ascii=use_ascii)
    summary = f'{left_repr} {op} {right_repr}'
    highlighter = config.get_terminal_writer()._highlight
    explanation = None
    try:
        if op == '==':
            explanation = _compare_eq_any(left, right, highlighter, verbose)
        elif op == 'not in':
            if istext(left) and istext(right):
                explanation = _notin_text(left, right, verbose)
        elif op == '!=':
            if isset(left) and isset(right):
                explanation = ['Both sets are equal']
        elif op == '>=':
            if isset(left) and isset(right):
                explanation = _compare_gte_set(left, right, highlighter, verbose)
        elif op == '<=':
            if isset(left) and isset(right):
                explanation = _compare_lte_set(left, right, highlighter, verbose)
        elif op == '>':
            if isset(left) and isset(right):
                explanation = _compare_gt_set(left, right, highlighter, verbose)
        elif op == '<':
            if isset(left) and isset(right):
                explanation = _compare_lt_set(left, right, highlighter, verbose)
    except outcomes.Exit:
        raise
    except Exception:
        explanation = ['(pytest_assertion plugin: representation of details failed: {}.'.format(_pytest._code.ExceptionInfo.from_current()._getreprcrash()), ' Probably an object has a faulty __repr__.)']
    if not explanation:
        return None
    if explanation[0] != '':
        explanation = ['', *explanation]
    return [summary, *explanation]