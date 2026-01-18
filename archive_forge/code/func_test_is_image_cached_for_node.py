from oslo_config import cfg
from oslo_db import options
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context as glance_context
import glance.db.sqlalchemy.api
from glance.db.sqlalchemy import models as db_models
from glance.db.sqlalchemy import models_metadef as metadef_models
import glance.tests.functional.db as db_tests
from glance.tests.functional.db import base
from glance.tests.functional.db import base_metadef
def test_is_image_cached_for_node(self):
    self.assertTrue(self.db_api.is_image_cached_for_node(self.adm_context, 'node_url_1', self.images[0]['id']))
    self.assertFalse(self.db_api.is_image_cached_for_node(self.adm_context, 'node_url_2', self.images[0]['id']))