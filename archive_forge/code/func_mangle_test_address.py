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
def mangle_test_address(address: str) -> List[str]:
    path, possible_open_bracket, params = address.partition('[')
    names = path.split('::')
    names[0] = names[0].replace(nodes.SEP, '.')
    names[0] = re.sub('\\.py$', '', names[0])
    names[-1] += possible_open_bracket + params
    return names