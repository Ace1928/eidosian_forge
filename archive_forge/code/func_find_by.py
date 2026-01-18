from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models.execution_environments import ExecutionEnvironment
from awx.main.models.jobs import JobTemplate
from awx.main.tests.functional.conftest import user, system_auditor  # noqa: F401; pylint: disable=unused-import
def find_by(result, name, key, value):
    for c in result[name]:
        if c[key] == value:
            return c
    values = [c.get(key, None) for c in result[name]]
    raise ValueError(f"Failed to find assets['{name}'][{key}] = '{value}' valid values are {values}")