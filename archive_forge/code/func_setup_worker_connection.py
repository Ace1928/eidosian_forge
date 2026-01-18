import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def setup_worker_connection(self, _worker_id):
    settings_dict = self.get_test_db_clone_settings(str(_worker_id))
    self.connection.settings_dict.update(settings_dict)
    self.connection.close()