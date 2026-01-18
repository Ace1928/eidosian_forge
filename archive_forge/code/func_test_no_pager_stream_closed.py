import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_no_pager_stream_closed(self) -> None:
    flush = mock.MagicMock(side_effect=ValueError)
    with sinks.BufferFixture() as out:
        with autopage.AutoPager(out.stream) as stream:
            stream.write('foo')
            stream.close()
            stream.flush = flush
        self.assertTrue(out.stream.closed)