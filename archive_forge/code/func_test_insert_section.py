from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
def test_insert_section(self):
    self.report.add_section(BasicView(), lambda: {'a': 1})
    self.report.add_section(BasicView(), basic_generator, 0)
    self.assertEqual(len(self.report.sections), 2)
    self.assertEqual(self.report.sections[0].generator, basic_generator)