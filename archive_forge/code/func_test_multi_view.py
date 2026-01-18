import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_multi_view(self):
    attached_view = text_generic.MultiView()
    self.model = base_model.ReportModel(data={}, attached_view=attached_view)
    self.model['1'] = mwdv_generator()
    self.model['2'] = mwdv_generator()
    self.model['2']['int'] = 2
    self.model.set_current_view_type('text')
    target_str = 'int = 1\nstring = value\nint = 2\nstring = value'
    self.assertEqual(target_str, str(self.model))