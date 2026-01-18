import pytest
import pyviz_comms
from pyviz_comms import extension
def test_get_ipython_fixture_reset(get_ipython):
    assert extension._last_execution_count == LAST_EXECUTION_COUNT
    assert extension._repeat_execution_in_cell == REPEAT_EXECUTION_IN_CELL