from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
def test_data_as_sequence_is_handled_properly(self):
    model = base_model.ReportModel(data=['a', 'b'])
    model.attached_view = BasicView()
    self.assertEqual('0: a;1: b;', str(model))