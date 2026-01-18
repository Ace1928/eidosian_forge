import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_custom_indent_string(self):
    view = text_generic.KeyValueView(indent_str='~~')
    self.model['lt'] = ['a', 'b']
    self.model.attached_view = view
    target_str = 'int = 1\nlt = \n~~a\n~~b\nstring = value'
    self.assertEqual(target_str, str(self.model))