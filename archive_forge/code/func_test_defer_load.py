import pytest
from panel import config, state
from panel.pane import HTML, panel
from panel.param import ParamFunction
from panel.tests.conftest import set_env_var
@pytest.mark.usefixtures('with_curdoc')
def test_defer_load():
    try:
        defer_load_old = config.defer_load
        config.defer_load = True

        def test():
            return 1
        assert ParamFunction.applies(test)
        assert isinstance(panel(test), ParamFunction)
    finally:
        config.defer_load = defer_load_old