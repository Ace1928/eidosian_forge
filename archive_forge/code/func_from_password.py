import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
@classmethod
def from_password(cls, host, username, password, insecure=None, jobs_api_version=None):
    return DatabricksConfig(host=host, username=username, password=password, token=None, refresh_token=None, insecure=insecure, jobs_api_version=jobs_api_version)