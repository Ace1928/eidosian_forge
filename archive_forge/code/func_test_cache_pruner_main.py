import io
import sys
from unittest import mock
import glance_store as store
from oslo_config import cfg
from oslo_log import log as logging
import glance.cmd.api
import glance.cmd.cache_cleaner
import glance.cmd.cache_pruner
import glance.common.config
from glance.common import exception as exc
import glance.common.wsgi
import glance.image_cache.cleaner
from glance.image_cache import prefetcher
import glance.image_cache.pruner
from glance.tests import utils as test_utils
@mock.patch.object(glance.common.config, 'parse_cache_args')
@mock.patch.object(logging, 'setup')
@mock.patch.object(glance.image_cache.ImageCache, 'init_driver')
@mock.patch.object(glance.image_cache.ImageCache, 'prune')
def test_cache_pruner_main(self, mock_cache_prune, mock_cache_init_driver, mock_log_setup, mock_parse_config):
    mock_cache_init_driver.return_value = None
    manager = mock.MagicMock()
    manager.attach_mock(mock_log_setup, 'mock_log_setup')
    manager.attach_mock(mock_parse_config, 'mock_parse_config')
    manager.attach_mock(mock_cache_init_driver, 'mock_cache_init_driver')
    manager.attach_mock(mock_cache_prune, 'mock_cache_prune')
    glance.cmd.cache_pruner.main()
    expected_call_sequence = [mock.call.mock_parse_config(), mock.call.mock_log_setup(CONF, 'glance'), mock.call.mock_cache_init_driver(), mock.call.mock_cache_prune()]
    self.assertEqual(expected_call_sequence, manager.mock_calls)