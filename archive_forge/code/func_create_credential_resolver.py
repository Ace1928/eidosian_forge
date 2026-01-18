import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
def create_credential_resolver(session, cache=None, region_name=None):
    """Create a default credential resolver.

    This creates a pre-configured credential resolver
    that includes the default lookup chain for
    credentials.

    """
    profile_name = session.get_config_variable('profile') or 'default'
    metadata_timeout = session.get_config_variable('metadata_service_timeout')
    num_attempts = session.get_config_variable('metadata_service_num_attempts')
    disable_env_vars = session.instance_variables().get('profile') is not None
    imds_config = {'ec2_metadata_service_endpoint': session.get_config_variable('ec2_metadata_service_endpoint'), 'ec2_metadata_service_endpoint_mode': resolve_imds_endpoint_mode(session), 'ec2_credential_refresh_window': _DEFAULT_ADVISORY_REFRESH_TIMEOUT, 'ec2_metadata_v1_disabled': session.get_config_variable('ec2_metadata_v1_disabled')}
    if cache is None:
        cache = {}
    env_provider = EnvProvider()
    container_provider = ContainerProvider()
    instance_metadata_provider = InstanceMetadataProvider(iam_role_fetcher=InstanceMetadataFetcher(timeout=metadata_timeout, num_attempts=num_attempts, user_agent=session.user_agent(), config=imds_config))
    profile_provider_builder = ProfileProviderBuilder(session, cache=cache, region_name=region_name)
    assume_role_provider = AssumeRoleProvider(load_config=lambda: session.full_config, client_creator=_get_client_creator(session, region_name), cache=cache, profile_name=profile_name, credential_sourcer=CanonicalNameCredentialSourcer([env_provider, container_provider, instance_metadata_provider]), profile_provider_builder=profile_provider_builder)
    pre_profile = [env_provider, assume_role_provider]
    profile_providers = profile_provider_builder.providers(profile_name=profile_name, disable_env_vars=disable_env_vars)
    post_profile = [OriginalEC2Provider(), BotoProvider(), container_provider, instance_metadata_provider]
    providers = pre_profile + profile_providers + post_profile
    if disable_env_vars:
        providers.remove(env_provider)
        logger.debug('Skipping environment variable credential check because profile name was explicitly set.')
    resolver = CredentialResolver(providers=providers)
    return resolver