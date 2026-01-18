from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
def get_method_spec(method):
    """Get a stable and compatible method spec.

    Newer features in Python3 (kw-only arguments and annotations) are
    not supported or representable with inspect.getargspec() but many
    object hashes are already recorded using that method. This attempts
    to return something compatible with getargspec() when possible (i.e.
    when those features are not used), and otherwise just returns the
    newer getfullargspec() representation.
    """
    fullspec = inspect.getfullargspec(method)
    if any([fullspec.kwonlyargs, fullspec.kwonlydefaults, fullspec.annotations]):
        return fullspec
    else:
        return CompatArgSpec(fullspec.args, fullspec.varargs, fullspec.varkw, fullspec.defaults)