import dataclasses
from io import StringIO
import os
from pprint import pprint
from typing import Any
from typing import cast
from typing import Dict
from typing import final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprEntry
from _pytest._code.code import ReprEntryNative
from _pytest._code.code import ReprExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import ReprFuncArgs
from _pytest._code.code import ReprLocals
from _pytest._code.code import ReprTraceback
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.config import Config
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import skip
def serialize_exception_longrepr(rep: BaseReport) -> Dict[str, Any]:
    assert rep.longrepr is not None
    longrepr = cast(ExceptionRepr, rep.longrepr)
    result: Dict[str, Any] = {'reprcrash': serialize_repr_crash(longrepr.reprcrash), 'reprtraceback': serialize_repr_traceback(longrepr.reprtraceback), 'sections': longrepr.sections}
    if isinstance(longrepr, ExceptionChainRepr):
        result['chain'] = []
        for repr_traceback, repr_crash, description in longrepr.chain:
            result['chain'].append((serialize_repr_traceback(repr_traceback), serialize_repr_crash(repr_crash), description))
    else:
        result['chain'] = None
    return result