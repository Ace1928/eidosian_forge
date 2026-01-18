from __future__ import annotations
from contextlib import (
import re
from typing import (
import warnings
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def deprecate_option(key: str, msg: str | None=None, rkey: str | None=None, removal_ver: str | None=None) -> None:
    """
    Mark option `key` as deprecated, if code attempts to access this option,
    a warning will be produced, using `msg` if given, or a default message
    if not.
    if `rkey` is given, any access to the key will be re-routed to `rkey`.

    Neither the existence of `key` nor that if `rkey` is checked. If they
    do not exist, any subsequence access will fail as usual, after the
    deprecation warning is given.

    Parameters
    ----------
    key : str
        Name of the option to be deprecated.
        must be a fully-qualified option name (e.g "x.y.z.rkey").
    msg : str, optional
        Warning message to output when the key is referenced.
        if no message is given a default message will be emitted.
    rkey : str, optional
        Name of an option to reroute access to.
        If specified, any referenced `key` will be
        re-routed to `rkey` including set/get/reset.
        rkey must be a fully-qualified option name (e.g "x.y.z.rkey").
        used by the default message if no `msg` is specified.
    removal_ver : str, optional
        Specifies the version in which this option will
        be removed. used by the default message if no `msg` is specified.

    Raises
    ------
    OptionError
        If the specified key has already been deprecated.
    """
    key = key.lower()
    if key in _deprecated_options:
        raise OptionError(f"Option '{key}' has already been defined as deprecated.")
    _deprecated_options[key] = DeprecatedOption(key, msg, rkey, removal_ver)