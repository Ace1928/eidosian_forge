from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
from oslo_i18n import _lazy
from oslo_i18n import _message
def test_log_level_domain_name(self):
    with mock.patch.object(_factory.TranslatorFactory, '_make_translation_func') as mtf:
        tf = _factory.TranslatorFactory('domain')
        tf._make_log_translation_func('mylevel')
        mtf.assert_called_with('domain-log-mylevel')