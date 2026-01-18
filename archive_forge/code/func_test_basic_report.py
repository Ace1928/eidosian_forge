import datetime
import io
import os
import re
import signal
import sys
import threading
from unittest import mock
import fixtures
import greenlet
from oslotest import base
import oslo_config
from oslo_config import fixture
from oslo_reports import guru_meditation_report as gmr
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import opts
def test_basic_report(self):
    report_lines = self.report.run().split('\n')
    target_str_header = ['========================================================================', '====                        Guru Meditation                         ====', '========================================================================', '||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||', '', '', '========================================================================', '====                            Package                             ====', '========================================================================', 'product = Sharp Cheddar', 'vendor = Cheese Shoppe', 'version = 1.0.0', '========================================================================', '====                            Threads                             ====', '========================================================================']
    self.assertEqual(target_str_header, report_lines[0:len(target_str_header)])
    self.assertTrue(re.match('------(\\s+)Thread #-?\\d+\\1\\s?------', report_lines[len(target_str_header)]))
    self.assertEqual('', report_lines[len(target_str_header) + 1])
    curr_line = skip_body_lines(len(target_str_header) + 2, report_lines)
    target_str_gt = ['========================================================================', '====                         Green Threads                          ====', '========================================================================', '------                        Green Thread                        ------', '']
    end_bound = curr_line + len(target_str_gt)
    self.assertEqual(target_str_gt, report_lines[curr_line:end_bound])
    curr_line = skip_body_lines(curr_line + len(target_str_gt), report_lines)
    target_str_p_head = ['========================================================================', '====                           Processes                            ====', '========================================================================']
    end_bound = curr_line + len(target_str_p_head)
    self.assertEqual(target_str_p_head, report_lines[curr_line:end_bound])
    curr_line += len(target_str_p_head)
    self.assertTrue(re.match('Process \\d+ \\(under \\d+\\)', report_lines[curr_line]))
    curr_line = skip_body_lines(curr_line + 1, report_lines)
    target_str_config = ['========================================================================', '====                         Configuration                          ====', '========================================================================', '']
    end_bound = curr_line + len(target_str_config)
    self.assertEqual(target_str_config, report_lines[curr_line:end_bound])