import argparse
import base64
import contextlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from oslo_utils import strutils
import yaml
from ironicclient.common.i18n import _
from ironicclient import exc
def bool_argument_value(arg_name, bool_str, strict=True, default=False):
    """Returns the Boolean represented by bool_str.

    Returns the Boolean value for the argument named arg_name. The value is
    represented by the string bool_str. If the string is an invalid Boolean
    string: if strict is True, a CommandError exception is raised; otherwise
    the default value is returned.

    :param arg_name: The name of the argument
    :param bool_str: The string representing a Boolean value
    :param strict: Used if the string is invalid. If True, raises an exception.
        If False, returns the default value.
    :param default: The default value to return if the string is invalid
        and not strict
    :returns: the Boolean value represented by bool_str or the default value
        if bool_str is invalid and strict is False
    :raises CommandError: if bool_str is an invalid Boolean string

    """
    try:
        val = strutils.bool_from_string(bool_str, strict, default)
    except ValueError as e:
        raise exc.CommandError(_('argument %(arg)s: %(err)s.') % {'arg': arg_name, 'err': e})
    return val