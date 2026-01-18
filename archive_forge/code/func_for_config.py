import dataclasses
import json
import os
from pathlib import Path
from typing import Dict
from typing import final
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Union
from .pathlib import resolve_from_str
from .pathlib import rm_rf
from .reports import CollectReport
from _pytest import nodes
from _pytest._io import TerminalWriter
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.nodes import Directory
from _pytest.nodes import File
from _pytest.reports import TestReport
@classmethod
def for_config(cls, config: Config, *, _ispytest: bool=False) -> 'Cache':
    """Create the Cache instance for a Config.

        :meta private:
        """
    check_ispytest(_ispytest)
    cachedir = cls.cache_dir_from_config(config, _ispytest=True)
    if config.getoption('cacheclear') and cachedir.is_dir():
        cls.clear_cache(cachedir, _ispytest=True)
    return cls(cachedir, config, _ispytest=True)