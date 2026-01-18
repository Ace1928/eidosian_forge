import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_line_buffering_on(self) -> None:
    ap = autopage.AutoPager(self.stream, line_buffering=True)
    ap._reconfigure_output_stream()
    self.addCleanup(ap._out.close)
    self.assertTrue(ap._out.line_buffering)
    self.assertEqual(self.default_errors, ap._out.errors)
    self.assertEqual(self.encoding, ap._out.encoding)
    self.assertIs(True, ap._line_buffering())
    self.assertEqual(self.default_errors, ap._errors())