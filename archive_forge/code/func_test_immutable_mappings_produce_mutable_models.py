from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
def test_immutable_mappings_produce_mutable_models(self):

    class SomeImmutableMapping(abc.Mapping):

        def __init__(self):
            self.data = {'a': 2, 'b': 4, 'c': 8}

        def __getitem__(self, key):
            return self.data[key]

        def __len__(self):
            return len(self.data)

        def __iter__(self):
            return iter(self.data)
    mp = SomeImmutableMapping()
    model = base_model.ReportModel(data=mp)
    model.attached_view = BasicView()
    self.assertEqual('a: 2;b: 4;c: 8;', str(model))
    model['d'] = 16
    self.assertEqual('a: 2;b: 4;c: 8;d: 16;', str(model))
    self.assertEqual({'a': 2, 'b': 4, 'c': 8}, mp.data)