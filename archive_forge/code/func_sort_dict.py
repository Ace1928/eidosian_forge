import os
import uuid
import xmltodict
from pytest import skip, fixture
from mock import patch
def sort_dict(ordered_dict):
    items = sorted(ordered_dict.items(), key=lambda x: x[0])
    ordered_dict.clear()
    for key, value in items:
        if isinstance(value, dict):
            sort_dict(value)
        ordered_dict[key] = value