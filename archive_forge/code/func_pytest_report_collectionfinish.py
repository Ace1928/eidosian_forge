from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from _pytest import nodes
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.reports import TestReport
import pytest
def pytest_report_collectionfinish(self) -> Optional[str]:
    if self.config.getoption('verbose') >= 0 and self.report_status:
        return f'stepwise: {self.report_status}'
    return None