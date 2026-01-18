import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
def decode_results(results):
    if not results:
        return []
    new_results = []
    for data, failures in results:
        new_failures = {}
        for key, data in failures.items():
            new_failures[key] = ft.Failure.from_dict(data)
        new_results.append((data, new_failures))
    return new_results