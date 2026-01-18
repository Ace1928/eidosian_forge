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
def parse_file_contents(data):
    """Parse the raw contents of a policy file.

    Parses the contents of a policy file which currently can be in either
    yaml or json format. Both can be parsed as yaml.

    :param data: A string containing the contents of a policy file.
    :returns: A dict of the form ``{'policy_name1': 'policy1',
        'policy_name2': 'policy2,...}``
    """
    try:
        parsed = jsonutils.loads(data)
        LOG.warning(WARN_JSON)
    except ValueError:
        try:
            parsed = yaml.safe_load(data)
        except yaml.YAMLError as e:
            raise ValueError(str(e))
    return parsed or {}