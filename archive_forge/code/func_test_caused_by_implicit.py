import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def test_caused_by_implicit(self):

    def raises_chained():
        try:
            raise Fail2('I have been broken')
        except Fail2:
            excutils.raise_with_cause(Fail1, 'I was broken')
    e = self.assertRaises(Fail1, raises_chained)
    self.assertIsInstance(e.cause, Fail2)
    e_p = e.pformat()
    self.assertIn('I have been broken', e_p)
    self.assertIn('Fail2', e_p)