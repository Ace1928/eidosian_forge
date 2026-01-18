import os
import textwrap
from optparse import Values
from typing import Any, List
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, PipError
from pip._internal.utils import filesystem
from pip._internal.utils.logging import getLogger
def remove_cache_items(self, options: Values, args: List[Any]) -> None:
    if len(args) > 1:
        raise CommandError('Too many arguments')
    if not args:
        raise CommandError('Please provide a pattern')
    files = self._find_wheels(options, args[0])
    no_matching_msg = 'No matching packages'
    if args[0] == '*':
        files += self._find_http_files(options)
    else:
        no_matching_msg += f' for pattern "{args[0]}"'
    if not files:
        logger.warning(no_matching_msg)
    for filename in files:
        os.unlink(filename)
        logger.verbose('Removed %s', filename)
    logger.info('Files removed: %s', len(files))