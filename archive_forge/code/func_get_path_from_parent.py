import bisect
import copy
import inspect
import warnings
from collections import defaultdict
from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db import connections
from django.db.models import AutoField, Manager, OrderWrt, UniqueConstraint
from django.db.models.query_utils import PathInfo
from django.utils.datastructures import ImmutableList, OrderedSet
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.text import camel_case_to_spaces, format_lazy
from django.utils.translation import override
def get_path_from_parent(self, parent):
    """
        Return a list of PathInfos containing the path from the parent
        model to the current model, or an empty list if parent is not a
        parent of the current model.
        """
    if self.model is parent:
        return []
    model = self.concrete_model
    chain = model._meta.get_base_chain(parent)
    chain.reverse()
    chain.append(model)
    path = []
    for i, ancestor in enumerate(chain[:-1]):
        child = chain[i + 1]
        link = child._meta.get_ancestor_link(ancestor)
        path.extend(link.reverse_path_infos)
    return path