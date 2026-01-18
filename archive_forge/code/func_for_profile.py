import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
@staticmethod
def for_profile(profile):
    if profile is None:
        return InvalidConfigurationError(f"You haven't configured the CLI yet! Please configure by entering `{sys.argv[0]} configure`")
    return InvalidConfigurationError(f"You haven't configured the CLI yet for the profile {profile}! Please configure by entering `{sys.argv[0]} configure --profile {profile}`")