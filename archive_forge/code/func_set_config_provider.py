import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
def set_config_provider(provider):
    """
    Sets a DatabricksConfigProvider that will be used for all future calls to get_config(),
    used by the Databricks CLI code to discover the user's credentials.
    """
    global _config_provider
    if provider and (not isinstance(provider, DatabricksConfigProvider)):
        raise Exception('Must be instance of DatabricksConfigProvider: %s' % _config_provider)
    _config_provider = provider