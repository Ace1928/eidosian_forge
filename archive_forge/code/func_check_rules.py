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
def check_rules(self, raise_on_violation=False):
    """Look for rule definitions that are obviously incorrect."""
    undefined_checks = []
    cyclic_checks = []
    violation = False
    for name, check in self.rules.items():
        if not self.skip_undefined_check and self._undefined_check(check):
            undefined_checks.append(name)
            violation = True
        if self._cycle_check(check):
            cyclic_checks.append(name)
            violation = True
    if undefined_checks:
        LOG.warning('Policies %(names)s reference a rule that is not defined.', {'names': undefined_checks})
    if cyclic_checks:
        LOG.warning('Policies %(names)s are part of a cyclical reference.', {'names': cyclic_checks})
    if raise_on_violation and violation:
        raise InvalidDefinitionError(undefined_checks + cyclic_checks)
    return not violation