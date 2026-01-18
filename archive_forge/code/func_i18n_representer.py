import collections
import copy
import importlib.metadata
import json
import logging
import operator
import sys
import yaml
from oslo_config import cfg
from oslo_i18n import _message
import stevedore.named  # noqa
def i18n_representer(dumper, data):
    """oslo_i18n yaml representer

    Returns a translated to the default locale string for yaml.safe_dump

    :param dumper: a SafeDumper instance passed by yaml.safe_dump
    :param data: a oslo_i18n._message.Message instance
    """
    serializedData = str(data.translation())
    return dumper.represent_str(serializedData)