from django.apps import apps
from django.contrib.admin.exceptions import NotRegistered
from django.core.exceptions import FieldDoesNotExist, PermissionDenied
from django.http import Http404, JsonResponse
from django.views.generic.list import BaseListView
def serialize_result(self, obj, to_field_name):
    """
        Convert the provided model object to a dictionary that is added to the
        results list.
        """
    return {'id': str(getattr(obj, to_field_name)), 'text': str(obj)}