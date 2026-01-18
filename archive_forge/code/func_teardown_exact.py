import bdb
import dataclasses
import os
import sys
from typing import Callable
from typing import cast
from typing import Dict
from typing import final
from typing import Generic
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .reports import BaseReport
from .reports import CollectErrorRepr
from .reports import CollectReport
from .reports import TestReport
from _pytest import timing
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.nodes import Collector
from _pytest.nodes import Directory
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.outcomes import Exit
from _pytest.outcomes import OutcomeException
from _pytest.outcomes import Skipped
from _pytest.outcomes import TEST_OUTCOME
def teardown_exact(self, nextitem: Optional[Item]) -> None:
    """Teardown the current stack up until reaching nodes that nextitem
        also descends from.

        When nextitem is None (meaning we're at the last item), the entire
        stack is torn down.
        """
    needed_collectors = nextitem and nextitem.listchain() or []
    exceptions: List[BaseException] = []
    while self.stack:
        if list(self.stack.keys()) == needed_collectors[:len(self.stack)]:
            break
        node, (finalizers, _) = self.stack.popitem()
        these_exceptions = []
        while finalizers:
            fin = finalizers.pop()
            try:
                fin()
            except TEST_OUTCOME as e:
                these_exceptions.append(e)
        if len(these_exceptions) == 1:
            exceptions.extend(these_exceptions)
        elif these_exceptions:
            msg = f'errors while tearing down {node!r}'
            exceptions.append(BaseExceptionGroup(msg, these_exceptions[::-1]))
    if len(exceptions) == 1:
        raise exceptions[0]
    elif exceptions:
        raise BaseExceptionGroup('errors during test teardown', exceptions[::-1])
    if nextitem is None:
        assert not self.stack