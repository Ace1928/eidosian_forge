import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_global_CommandLine_output(tmpdir):
    """Ensures CommandLine.set_default_terminal_output works"""
    from nipype.interfaces.fsl import BET
    ci = nib.CommandLine(command='ls -l')
    assert ci.terminal_output == 'stream'
    ci = BET()
    assert ci.terminal_output == 'stream'
    with mock.patch.object(nib.CommandLine, '_terminal_output'):
        nib.CommandLine.set_default_terminal_output('allatonce')
        ci = nib.CommandLine(command='ls -l')
        assert ci.terminal_output == 'allatonce'
        nib.CommandLine.set_default_terminal_output('file')
        ci = nib.CommandLine(command='ls -l')
        assert ci.terminal_output == 'file'
        ci = BET()
        assert ci.terminal_output == 'file'