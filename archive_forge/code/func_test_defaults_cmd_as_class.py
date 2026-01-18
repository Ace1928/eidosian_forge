import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_defaults_cmd_as_class(self) -> None:

    class TestCommand(command.PagerCommand):

        def command(self) -> List[str]:
            return []

        def environment_variables(self, config: _PagerConfig) -> Optional[Dict[str, str]]:
            return None
    with mock.patch.object(TestCommand, 'command') as cmd:
        ap = autopage.AutoPager(pager_command=TestCommand, line_buffering=False)
        with mock.patch.object(ap, '_pager_env') as get_env:
            stream = ap._paged_stream()
            self.popen.assert_called_once_with(cmd.return_value, env=get_env.return_value, bufsize=-1, universal_newlines=True, encoding='UTF-8', errors='strict', stdin=subprocess.PIPE, stdout=None)
            self.assertIs(stream, self.popen.return_value.stdin)