import copy
from unittest import mock
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import report
from oslo_reports.views import jinja_view as jv
from oslo_reports.views.json import generic as json_generic
from oslo_reports.views.text import generic as text_generic
def test_list_in_dict_serialization(self):
    self.model['dt'] = {'a': 1, 'b': [2, 3]}
    target_str = 'dt = \n  a = 1\n  b = \n    2\n    3\nint = 1\nstring = value'
    self.assertEqual(target_str, str(self.model))