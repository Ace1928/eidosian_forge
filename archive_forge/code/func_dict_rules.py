import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
def dict_rules(enforcer_rules):
    """Converts enforcer rules to dictionary.

            :param enforcer_rules: enforcer rules represented as a class Rules
            :return: enforcer rules represented as a dictionary
            """
    return jsonutils.loads(str(enforcer_rules))