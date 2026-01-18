import io
import logging
import os
from shlex import split as shsplit
import sys
import numpy
def lint_cfg(cfgp, **paths):
    if not paths:
        paths = get_paths_cfg()
    cfgp_ref = ConfigParser()
    cfgp_ref.read([paths['sys'], paths['platform']])
    for loc, path in paths.items():
        exists = os.path.exists(path)
        msg = ' '.join(['{} file'.format(loc).rjust(13), 'exists:' if exists else 'does not exist:', path])
        logger.info(msg) if exists else logger.warning(msg)
    for section in cfgp.sections():
        if cfgp_ref.has_section(section):
            options = set(cfgp.options(section))
            options_ref = set(cfgp_ref.options(section))
            if options.issubset(options_ref):
                logger.info('pythranrc section [{}] is valid and options are correct'.format(section))
            else:
                logger.warning('pythranrc section [{}] is valid but options {} are incorrect!'.format(section, options.difference(options_ref)))
        else:
            logger.warning('pythranrc section [{}] is invalid!'.format(section))