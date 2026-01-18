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
def pick_default_policy_file(conf, fallback_to_json_file=True):
    new_default_policy_file = 'policy.yaml'
    old_default_policy_file = 'policy.json'
    policy_file = None
    if conf.oslo_policy.policy_file == new_default_policy_file and fallback_to_json_file:
        location = conf.get_location('policy_file', 'oslo_policy').location
        if conf.find_file(conf.oslo_policy.policy_file):
            policy_file = conf.oslo_policy.policy_file
        elif location in [cfg.Locations.opt_default, cfg.Locations.set_default]:
            LOG.debug('Searching old policy.json file.')
            if conf.find_file(old_default_policy_file):
                policy_file = old_default_policy_file
        if policy_file:
            LOG.debug('Picking default policy file: %s. Config location: %s', policy_file, location)
            return policy_file
    LOG.debug('No default policy file present, picking the configured one: %s.', conf.oslo_policy.policy_file)
    return conf.oslo_policy.policy_file