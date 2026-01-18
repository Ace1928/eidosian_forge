import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_basic_kv_view(self):
    attached_view = text_generic.BasicKeyValueView()
    self.model = base_model.ReportModel(data={'string': 'value', 'int': 1}, attached_view=attached_view)
    self.assertEqual('int = 1\nstring = value\n', str(self.model))