import functools
import logging
import os
import pkgutil
import re
import traceback
from oslo_utils import strutils
from zunclient import exceptions
from zunclient.i18n import _
def check_major_version(api_version):
    """Checks major part of ``APIVersion`` obj is supported.

    :raises exceptions.UnsupportedVersion: if major part is not supported
    """
    available_versions = get_available_major_versions()
    if not api_version.is_null() and str(api_version.ver_major) not in available_versions:
        if len(available_versions) == 1:
            msg = _("Invalid client version '%(version)s'. Major part should be '%(major)s'") % {'version': api_version.get_string(), 'major': available_versions[0]}
        else:
            msg = _("Invalid client version '%(version)s'. Major part must be one of: '%(major)s'") % {'version': api_version.get_string(), 'major': ', '.join(available_versions)}
        raise exceptions.UnsupportedVersion(msg)