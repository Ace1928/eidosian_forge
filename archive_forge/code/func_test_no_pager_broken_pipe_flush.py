import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_no_pager_broken_pipe_flush(self) -> None:
    flush = mock.MagicMock(side_effect=BrokenPipeError)
    with sinks.BufferFixture() as out:
        with autopage.AutoPager(out.stream) as stream:
            stream.write('foo')
            stream.flush = flush
        self.assertTrue(out.stream.closed)
    flush.assert_called_once()