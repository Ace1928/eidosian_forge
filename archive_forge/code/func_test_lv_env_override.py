import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_lv_env_override(self) -> None:
    config = command.PagerConfig(color=True, line_buffering_requested=False, reset_terminal=False)
    with fixtures.EnvironmentVariable('LV', 'abc'):
        env = self._env(config)
    self.assertNotIn('LV', env)