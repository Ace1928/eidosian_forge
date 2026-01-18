import pytest
from panel.depends import bind, transform_reference
from panel.pane import panel
from panel.param import ParamFunction
from panel.widgets import IntSlider
def test_bind_bare_emits_warning(caplog):

    def foo():
        return 'bar'
    ParamFunction(foo)
    panel(bind(foo))
    found = False
    for log_record in caplog.records:
        if log_record.levelname == 'WARNING' and "The function 'foo' does not have any dependencies and will never update" in log_record.message:
            found = True
    assert found