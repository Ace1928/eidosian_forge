import collections.abc
import copy
import logging
import os
import typing as ty
import warnings
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import strutils
import yaml
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_policy import _parser
from oslo_policy import opts
def register_default(self, default):
    """Registers a RuleDefault.

        Adds a RuleDefault to the list of registered rules. Rules must be
        registered before using the Enforcer.authorize method.

        :param default: A RuleDefault object to register.
        """
    if default.name in self.registered_rules:
        raise DuplicatePolicyError(default.name)
    self.registered_rules[default.name] = copy.deepcopy(default)