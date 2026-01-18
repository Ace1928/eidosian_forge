import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
def env_as_dict(self):
    """Get the entire environment as a dict."""
    user_env = self.user_env_as_dict()
    user_env.update({env_fmt.ENCRYPTED_PARAM_NAMES: self.encrypted_param_names})
    return user_env