import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
@classmethod
def generate_scenarios(cls):
    cls.scenarios = testscenarios.multiply_scenarios(cls._unit_system, cls._sign, cls._magnitude, cls._unit_prefix, cls._unit_suffix, cls._return_int)