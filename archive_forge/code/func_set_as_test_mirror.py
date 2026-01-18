import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def set_as_test_mirror(self, primary_settings_dict):
    """
        Set this database up to be used in testing as a mirror of a primary
        database whose settings are given.
        """
    self.connection.settings_dict['NAME'] = primary_settings_dict['NAME']