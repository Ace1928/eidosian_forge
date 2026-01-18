import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_CommandLine_prefix(tmpdir):
    tmpdir.chdir()
    oop = 'out/of/path'
    os.makedirs(oop)
    script_name = 'test_script.sh'
    script_path = os.path.join(oop, script_name)
    with open(script_path, 'w') as script_f:
        script_f.write('#!/usr/bin/env bash\necho Success!')
    os.chmod(script_path, 493)
    ci = nib.CommandLine(command=script_name)
    with pytest.raises(IOError):
        ci.run()

    class OOPCLI(nib.CommandLine):
        _cmd_prefix = oop + '/'
    ci = OOPCLI(command=script_name)
    ci.run()

    class OOPShell(nib.CommandLine):
        _cmd_prefix = 'bash {}/'.format(oop)
    ci = OOPShell(command=script_name)
    ci.run()

    class OOPBadShell(nib.CommandLine):
        _cmd_prefix = 'shell_dne {}/'.format(oop)
    ci = OOPBadShell(command=script_name)
    with pytest.raises(IOError):
        ci.run()