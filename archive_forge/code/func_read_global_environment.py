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
def read_global_environment(env, env_dir=None):
    if env_dir is None:
        cfg.CONF.import_opt('environment_dir', 'heat.common.config')
        env_dir = cfg.CONF.environment_dir
    try:
        env_files = glob.glob(os.path.join(env_dir, '*'))
    except OSError:
        LOG.exception('Failed to read %s', env_dir)
        return
    for file_path in env_files:
        try:
            with open(file_path) as env_fd:
                LOG.info('Loading %s', file_path)
                env_body = env_fmt.parse(env_fd.read())
                env_fmt.default_for_missing(env_body)
                env.load(env_body)
        except ValueError:
            LOG.exception('Failed to parse %s', file_path)
        except IOError:
            LOG.exception('Failed to read %s', file_path)