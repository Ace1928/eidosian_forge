import functools
import logging
import logging.config
import optparse
import os
import sys
import traceback
from optparse import Values
from typing import Any, Callable, List, Optional, Tuple
from pip._vendor.rich import traceback as rich_traceback
from pip._internal.cli import cmdoptions
from pip._internal.cli.command_context import CommandContextMixIn
from pip._internal.cli.parser import ConfigOptionParser, UpdatingDefaultsHelpFormatter
from pip._internal.cli.status_codes import (
from pip._internal.exceptions import (
from pip._internal.utils.filesystem import check_path_owner
from pip._internal.utils.logging import BrokenStdoutLoggingError, setup_logging
from pip._internal.utils.misc import get_prog, normalize_path
from pip._internal.utils.temp_dir import TempDirectoryTypeRegistry as TempDirRegistry
from pip._internal.utils.temp_dir import global_tempdir_manager, tempdir_registry
from pip._internal.utils.virtualenv import running_under_virtualenv
@functools.wraps(run_func)
def exc_logging_wrapper(*args: Any) -> int:
    try:
        status = run_func(*args)
        assert isinstance(status, int)
        return status
    except DiagnosticPipError as exc:
        logger.error('%s', exc, extra={'rich': True})
        logger.debug('Exception information:', exc_info=True)
        return ERROR
    except PreviousBuildDirError as exc:
        logger.critical(str(exc))
        logger.debug('Exception information:', exc_info=True)
        return PREVIOUS_BUILD_DIR_ERROR
    except (InstallationError, UninstallationError, BadCommand, NetworkConnectionError) as exc:
        logger.critical(str(exc))
        logger.debug('Exception information:', exc_info=True)
        return ERROR
    except CommandError as exc:
        logger.critical('%s', exc)
        logger.debug('Exception information:', exc_info=True)
        return ERROR
    except BrokenStdoutLoggingError:
        print('ERROR: Pipe to stdout was broken', file=sys.stderr)
        if level_number <= logging.DEBUG:
            traceback.print_exc(file=sys.stderr)
        return ERROR
    except KeyboardInterrupt:
        logger.critical('Operation cancelled by user')
        logger.debug('Exception information:', exc_info=True)
        return ERROR
    except BaseException:
        logger.critical('Exception:', exc_info=True)
        return UNKNOWN_ERROR