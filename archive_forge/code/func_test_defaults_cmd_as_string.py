import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_defaults_cmd_as_string(self) -> None:
    ap = autopage.AutoPager(pager_command='foo bar', line_buffering=False)
    with mock.patch.object(ap, '_pager_env') as get_env:
        stream = ap._paged_stream()
        self.popen.assert_called_once_with(['foo', 'bar'], env=get_env.return_value, bufsize=-1, universal_newlines=True, encoding='UTF-8', errors='strict', stdin=subprocess.PIPE, stdout=None)
        self.assertIs(stream, self.popen.return_value.stdin)