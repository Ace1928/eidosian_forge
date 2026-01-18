import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_Commandline_environ(monkeypatch, tmpdir):
    from nipype import config
    config.set_default_config()
    tmpdir.chdir()
    monkeypatch.setitem(os.environ, 'DISPLAY', ':1')
    ci3 = nib.CommandLine(command='echo')
    res = ci3.run()
    assert res.runtime.environ['DISPLAY'] == ':1'
    monkeypatch.delitem(os.environ, 'DISPLAY', raising=False)
    config.set('execution', 'display_variable', ':3')
    res = ci3.run()
    assert 'DISPLAY' not in ci3.inputs.environ
    assert 'DISPLAY' not in res.runtime.environ
    ci3._redirect_x = True
    res = ci3.run()
    assert res.runtime.environ['DISPLAY'] == ':3'
    monkeypatch.setitem(os.environ, 'DISPLAY', ':1')
    ci3.inputs.environ = {'DISPLAY': ':2'}
    res = ci3.run()
    assert res.runtime.environ['DISPLAY'] == ':2'