import os
import importlib
import pytest
from IPython.terminal.pt_inputhooks import set_qt_api, get_inputhook_name_and_func
@pytest.mark.skipif(len(guis_avail) == 0, reason='No viable version of PyQt or PySide installed.')
def test_inputhook_qt():
    gui_ret, _ = get_inputhook_name_and_func('qt')
    assert gui_ret != 'qt'
    assert gui_ret in guis_avail
    if len(guis_avail) > 2:
        for not_gui in ['qt6', 'qt5']:
            if not_gui != gui_ret:
                break
        gui_ret2, _ = get_inputhook_name_and_func(not_gui)
        assert gui_ret2 == gui_ret
        assert gui_ret2 != not_gui