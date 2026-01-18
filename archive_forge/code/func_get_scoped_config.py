import copy
import logging
import os
import platform
import socket
import warnings
import botocore.client
import botocore.configloader
import botocore.credentials
import botocore.tokens
from botocore import (
from botocore.compat import HAS_CRT, MutableMapping
from botocore.configprovider import (
from botocore.errorfactory import ClientExceptionsFactory
from botocore.exceptions import (
from botocore.hooks import (
from botocore.loaders import create_loader
from botocore.model import ServiceModel
from botocore.parsers import ResponseParserFactory
from botocore.regions import EndpointResolver
from botocore.useragent import UserAgentString
from botocore.utils import (
from botocore.compat import HAS_CRT  # noqa
def get_scoped_config(self):
    """
        Returns the config values from the config file scoped to the current
        profile.

        The configuration data is loaded **only** from the config file.
        It does not resolve variables based on different locations
        (e.g. first from the session instance, then from environment
        variables, then from the config file).  If you want this lookup
        behavior, use the ``get_config_variable`` method instead.

        Note that this configuration is specific to a single profile (the
        ``profile`` session variable).

        If the ``profile`` session variable is set and the profile does
        not exist in the config file, a ``ProfileNotFound`` exception
        will be raised.

        :raises: ConfigNotFound, ConfigParseError, ProfileNotFound
        :rtype: dict

        """
    profile_name = self.get_config_variable('profile')
    profile_map = self._build_profile_map()
    if profile_name is None:
        return profile_map.get('default', {})
    elif profile_name not in profile_map:
        raise ProfileNotFound(profile=profile_name)
    else:
        return profile_map[profile_name]