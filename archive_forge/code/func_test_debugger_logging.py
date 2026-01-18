import logging
import pytest
import panel as pn
@pytest.mark.xdist_group('debugger')
def test_debugger_logging():
    logger = logging.getLogger('panel.callbacks')
    debugger = pn.widgets.Debugger()
    logger.info('test')
    assert debugger.terminal.output == ''
    assert debugger.title == ''
    logger.error('error test')
    assert 'error test' in debugger.terminal.output
    assert 'errors: </span>1' in debugger.title
    debugger.terminal._clears += 1
    assert debugger.title == ''