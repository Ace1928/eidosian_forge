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
def build_image_fixture(**kwargs):
    default_datetime = timeutils.utcnow()
    image = {'id': str(uuid.uuid4()), 'name': 'fake image #2', 'status': 'active', 'disk_format': 'vhd', 'container_format': 'ovf', 'is_public': True, 'created_at': default_datetime, 'updated_at': default_datetime, 'deleted_at': None, 'deleted': False, 'checksum': None, 'min_disk': 5, 'min_ram': 256, 'size': 19, 'locations': [{'url': 'file:///tmp/glance-tests/2', 'metadata': {}, 'status': 'active'}], 'properties': {}}
    if 'visibility' in kwargs:
        image.pop('is_public')
    image.update(kwargs)
    return image