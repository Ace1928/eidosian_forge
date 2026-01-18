from typing import Any, Dict, Sequence
import _pytest
import pytest
from onnx.backend.test.report.coverage import Coverage
@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter: _pytest.terminal.TerminalReporter, exitstatus: int) -> None:
    for mark in _marks.values():
        _add_mark(mark, 'loaded')
    _coverage.report_text(terminalreporter)