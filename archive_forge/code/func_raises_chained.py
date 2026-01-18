import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
def raises_chained():
    try:
        raise Fail2('I have been broken')
    except Fail2:
        excutils.raise_with_cause(Fail1, 'I was broken')