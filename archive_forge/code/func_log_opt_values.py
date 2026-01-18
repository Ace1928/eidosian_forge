import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
def log_opt_values(self, logger, lvl):
    """Log the value of all registered opts.

        It's often useful for an app to log its configuration to a log file at
        startup for debugging. This method dumps to the entire config state to
        the supplied logger at a given log level.

        :param logger: a logging.Logger object
        :param lvl: the log level (for example logging.DEBUG) arg to
                    logger.log()
        """
    logger.log(lvl, '*' * 80)
    logger.log(lvl, 'Configuration options gathered from:')
    logger.log(lvl, 'command line args: %s', self._args)
    logger.log(lvl, 'config files: %s', hasattr(self, 'config_file') and self.config_file or [])
    logger.log(lvl, '=' * 80)

    def _sanitize(opt, value):
        """Obfuscate values of options declared secret."""
        return value if not opt.secret else '*' * 4
    for opt_name in sorted(self._opts):
        opt = self._get_opt_info(opt_name)['opt']
        logger.log(lvl, '%-30s = %s', opt_name, _sanitize(opt, getattr(self, opt_name)))
    for group_name in list(self._groups):
        group_attr = self.GroupAttr(self, self._get_group(group_name))
        for opt_name in sorted(self._groups[group_name]._opts):
            opt = self._get_opt_info(opt_name, group_name)['opt']
            logger.log(lvl, '%-30s = %s', '%s.%s' % (group_name, opt_name), _sanitize(opt, getattr(group_attr, opt_name)))
    logger.log(lvl, '*' * 80)