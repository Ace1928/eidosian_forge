from datetime import datetime
import functools
import os
import platform
import re
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import Optional
from typing import Tuple
from typing import Union
import xml.etree.ElementTree as ET
from _pytest import nodes
from _pytest import timing
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprFileLocation
from _pytest.config import Config
from _pytest.config import filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
def node_reporter(self, report: Union[TestReport, str]) -> _NodeReporter:
    nodeid: Union[str, TestReport] = getattr(report, 'nodeid', report)
    workernode = getattr(report, 'node', None)
    key = (nodeid, workernode)
    if key in self.node_reporters:
        return self.node_reporters[key]
    reporter = _NodeReporter(nodeid, self)
    self.node_reporters[key] = reporter
    self.node_reporters_ordered.append(reporter)
    return reporter