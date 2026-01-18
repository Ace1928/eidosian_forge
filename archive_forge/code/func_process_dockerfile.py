import json
import logging
import os
import random
from .. import auth
from .. import constants
from .. import errors
from .. import utils
def process_dockerfile(dockerfile, path):
    if not dockerfile:
        return (None, None)
    abs_dockerfile = dockerfile
    if not os.path.isabs(dockerfile):
        abs_dockerfile = os.path.join(path, dockerfile)
        if constants.IS_WINDOWS_PLATFORM and path.startswith(constants.WINDOWS_LONGPATH_PREFIX):
            abs_dockerfile = '{}{}'.format(constants.WINDOWS_LONGPATH_PREFIX, os.path.normpath(abs_dockerfile[len(constants.WINDOWS_LONGPATH_PREFIX):]))
    if os.path.splitdrive(path)[0] != os.path.splitdrive(abs_dockerfile)[0] or os.path.relpath(abs_dockerfile, path).startswith('..'):
        with open(abs_dockerfile) as df:
            return (f'.dockerfile.{random.getrandbits(160):x}', df.read())
    if dockerfile == abs_dockerfile:
        dockerfile = os.path.relpath(abs_dockerfile, path)
    return (dockerfile, None)