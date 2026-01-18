import unittest
import fixtures  # type: ignore
from typing import Any, Optional, Dict, List
import autopage
from autopage import command
def test_env_empty(self) -> None:
    cmd = self.TestCommand({})
    ap = autopage.AutoPager(pager_command=cmd)
    with fixtures.EnvironmentVariable('BAZ', 'quux'):
        env = ap._pager_env()
    self.assertIsNone(env)