from unittest import mock
from unittest.mock import patch
import uuid
import glance_store
from oslo_config import cfg
from glance.common import exception
from glance.db.sqlalchemy import api as db_api
from glance import scrubber
from glance.tests import utils as test_utils
def make_get_images_detailed(pager):

    def mock_get_images_detailed(ctx, filters, marker=None, limit=None):
        return pager()
    return mock_get_images_detailed