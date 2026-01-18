import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_list_shared_images_other_tenant(self):
    image5 = _db_fixture(uuids.image5, owner=TENANT3, name='5', size=512, is_public=False)
    self.db.image_create(None, image5)
    context = glance.context.RequestContext(user=USER1, tenant=TENANT3)
    image_repo = glance.db.ImageRepo(context, self.db)
    images = {i.image_id: i for i in image_repo.list()}
    self.assertIsNone(images[UUID1].member)
    self.assertEqual(TENANT3, images[UUID2].member)
    self.assertIsNone(images[uuids.image5].member)