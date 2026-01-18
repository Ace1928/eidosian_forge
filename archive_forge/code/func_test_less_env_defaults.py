import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_less_env_defaults(self) -> None:
    config = command.PagerConfig(color=True, line_buffering_requested=False, reset_terminal=False)
    less_env = self._env(config)['LESS']
    self.assertEqual('RFX', less_env)