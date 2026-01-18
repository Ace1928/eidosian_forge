import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def test_lv_env_nocolor_reset(self) -> None:
    config = command.PagerConfig(color=False, line_buffering_requested=False, reset_terminal=True)
    self.assertNotIn('LV', self._env(config))