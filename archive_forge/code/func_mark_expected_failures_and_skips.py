import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def mark_expected_failures_and_skips(self):
    """
        Mark tests in Django's test suite which are expected failures on this
        database and test which should be skipped on this database.
        """
    from unittest import expectedFailure, skip
    for test_name in self.connection.features.django_test_expected_failures:
        test_case_name, _, test_method_name = test_name.rpartition('.')
        test_app = test_name.split('.')[0]
        if test_app in settings.INSTALLED_APPS:
            test_case = import_string(test_case_name)
            test_method = getattr(test_case, test_method_name)
            setattr(test_case, test_method_name, expectedFailure(test_method))
    for reason, tests in self.connection.features.django_test_skips.items():
        for test_name in tests:
            test_case_name, _, test_method_name = test_name.rpartition('.')
            test_app = test_name.split('.')[0]
            if test_app in settings.INSTALLED_APPS:
                test_case = import_string(test_case_name)
                test_method = getattr(test_case, test_method_name)
                setattr(test_case, test_method_name, skip(reason)(test_method))