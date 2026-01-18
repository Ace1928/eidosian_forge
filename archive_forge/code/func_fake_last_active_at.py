import datetime
from unittest import mock
import uuid
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import sql_model as model
from keystone.identity.shadow_backends import sql as shadow_sql
from keystone.tests import unit
def fake_last_active_at(self, user_id):
    test_self._delete_user(user_id)
    real_last_active_at(self, user_id)