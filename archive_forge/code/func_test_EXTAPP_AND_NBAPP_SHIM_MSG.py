import io
import logging
import pytest
from traitlets import default
from .mockextension import MockExtensionApp
from notebook_shim import shim
@pytest.mark.parametrize('jp_argv,trait_name,trait_value', list_test_params([('enable_mathjax', False)]))
def test_EXTAPP_AND_NBAPP_SHIM_MSG(read_app_logs, extensionapp, jp_argv, trait_name, trait_value):
    log = read_app_logs()
    log_msg = shim.EXTAPP_AND_NBAPP_SHIM_MSG(trait_name, 'MockExtensionApp')
    assert log_msg in log
    assert getattr(extensionapp, trait_name) == trait_value