import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_dict_serialization_integer_keys(self):
    self.model['dt'] = {3: 4, 5: 6}
    target_str = 'dt = \n  3 = 4\n  5 = 6\nint = 1\nstring = value'
    self.assertEqual(target_str, str(self.model))