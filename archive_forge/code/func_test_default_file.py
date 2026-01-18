import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_default_file(self) -> None:
    with sinks.TempFixture() as out:
        with fixtures.MonkeyPatch('sys.stdout', out.stream):
            ap = autopage.AutoPager()
        self.assertFalse(ap.to_terminal())