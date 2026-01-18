import unittest
from apitools.base.py import list_pager
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 \
from samples.fusiontables_sample.fusiontables_v1 \
from samples.iam_sample.iam_v1 import iam_v1_client as iam_client
from samples.iam_sample.iam_v1 import iam_v1_messages as iam_messages
def testSetattrNested(self):
    o = Example()
    list_pager._SetattrNested(o, 'b', Example())
    self.assertEqual(o.b.a, 'aaa')
    list_pager._SetattrNested(o, ('b', 'a'), 'AAA')
    self.assertEqual(o.b.a, 'AAA')
    list_pager._SetattrNested(o, ('c',), 'CCC')
    self.assertEqual(o.c, 'CCC')