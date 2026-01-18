import pytest
from panel import config, state
from panel.pane import HTML, panel
from panel.param import ParamFunction
from panel.tests.conftest import set_env_var
def test_console_output_disable_stdout(document, comm, get_display_handle):
    pane = HTML()
    with set_env_var('PANEL_CONSOLE_OUTPUT', 'disable'):
        model = pane.get_root(document, comm)
        handle = get_display_handle(model)
        pane._on_stdout(model.ref['id'], ['print output'])
        assert handle == {}
        pane._cleanup(model)
        assert model.ref['id'] not in state._handles