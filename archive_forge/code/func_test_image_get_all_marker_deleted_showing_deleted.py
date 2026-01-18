import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_image_get_all_marker_deleted_showing_deleted(self):
    """Specify a deleted image as a marker if showing deleted images.

        A non-admin user has to explicitly ask for deleted
        images, and should only see deleted images in the results
        """
    self.db_api.image_destroy(self.adm_context, UUID3)
    self.db_api.image_destroy(self.adm_context, UUID1)
    filters = {'deleted': True}
    images = self.db_api.image_get_all(self.context, marker=UUID3, filters=filters)
    self.assertEqual(1, len(images))