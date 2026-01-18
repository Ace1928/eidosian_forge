from collections.abc import Mapping
import dataclasses
import os
import platform
import sys
import traceback
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Type
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.mark.structures import Mark
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.reports import BaseReport
from _pytest.reports import TestReport
from _pytest.runner import CallInfo
from _pytest.stash import StashKey
def evaluate_xfail_marks(item: Item) -> Optional[Xfail]:
    """Evaluate xfail marks on item, returning Xfail if triggered."""
    for mark in item.iter_markers(name='xfail'):
        run = mark.kwargs.get('run', True)
        strict = mark.kwargs.get('strict', item.config.getini('xfail_strict'))
        raises = mark.kwargs.get('raises', None)
        if 'condition' not in mark.kwargs:
            conditions = mark.args
        else:
            conditions = (mark.kwargs['condition'],)
        if not conditions:
            reason = mark.kwargs.get('reason', '')
            return Xfail(reason, run, strict, raises)
        for condition in conditions:
            result, reason = evaluate_condition(item, mark, condition)
            if result:
                return Xfail(reason, run, strict, raises)
    return None