import subprocess
import sys
import unittest
from unittest import mock
import fixtures  # type: ignore
from typing import Optional, List, Dict
from autopage.tests import sinks
import autopage
from autopage import command
def test_launch_pager_fail(self) -> None:
    outstream = mock.Mock()
    ap = autopage.AutoPager(outstream)
    with mock.patch.object(ap, 'to_terminal', return_value=True), mock.patch.object(ap, '_paged_stream', side_effect=OSError) as page, mock.patch.object(ap, '_reconfigure_output_stream') as reconf:
        with ap as stream:
            page.assert_called_once()
            reconf.assert_called_once()
            self.assertIs(outstream, stream)