from __future__ import annotations
import copy
import datetime as dt
import decimal
import inspect
import json
import typing
import uuid
import warnings
from abc import ABCMeta
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from marshmallow import base, class_registry, types
from marshmallow import fields as ma_fields
from marshmallow.decorators import (
from marshmallow.error_store import ErrorStore
from marshmallow.exceptions import StringNotCollectionError, ValidationError
from marshmallow.orderedset import OrderedSet
from marshmallow.utils import (
from marshmallow.warnings import RemovedInMarshmallow4Warning
@classmethod
def get_declared_fields(mcs, klass: type, cls_fields: list, inherited_fields: list, dict_cls: type=dict):
    """Returns a dictionary of field_name => `Field` pairs declared on the class.
        This is exposed mainly so that plugins can add additional fields, e.g. fields
        computed from class Meta options.

        :param klass: The class object.
        :param cls_fields: The fields declared on the class, including those added
            by the ``include`` class Meta option.
        :param inherited_fields: Inherited fields.
        :param dict_cls: dict-like class to use for dict output Default to ``dict``.
        """
    return dict_cls(inherited_fields + cls_fields)