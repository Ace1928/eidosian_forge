import pytest
import pyviz_comms
from pyviz_comms import extension
def test_extension_count_two_cells_one_extension(monkeypatch, get_ipython):
    monkeypatch.setattr(pyviz_comms, 'get_ipython', get_ipython)

    class sub_extension(extension):

        def __call__(self, *args, **params):
            pass
    sub_extension()
    get_ipython().bump()
    sub_extension()
    assert sub_extension._repeat_execution_in_cell is False
    assert sub_extension._repeat_execution_in_cell == extension._repeat_execution_in_cell
    sub_extension()
    assert sub_extension._repeat_execution_in_cell is True
    assert sub_extension._repeat_execution_in_cell == extension._repeat_execution_in_cell
    get_ipython().bump()
    sub_extension()
    assert sub_extension._repeat_execution_in_cell is False
    assert sub_extension._repeat_execution_in_cell == extension._repeat_execution_in_cell