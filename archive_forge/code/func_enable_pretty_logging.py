import logging
import logging.handlers
import sys
from tornado.escape import _unicode
from tornado.util import unicode_type, basestring_type
from typing import Dict, Any, cast, Optional
def enable_pretty_logging(options: Any=None, logger: Optional[logging.Logger]=None) -> None:
    """Turns on formatted logging output as configured.

    This is called automatically by `tornado.options.parse_command_line`
    and `tornado.options.parse_config_file`.
    """
    if options is None:
        import tornado.options
        options = tornado.options.options
    if options.logging is None or options.logging.lower() == 'none':
        return
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(getattr(logging, options.logging.upper()))
    if options.log_file_prefix:
        rotate_mode = options.log_rotate_mode
        if rotate_mode == 'size':
            channel = logging.handlers.RotatingFileHandler(filename=options.log_file_prefix, maxBytes=options.log_file_max_size, backupCount=options.log_file_num_backups, encoding='utf-8')
        elif rotate_mode == 'time':
            channel = logging.handlers.TimedRotatingFileHandler(filename=options.log_file_prefix, when=options.log_rotate_when, interval=options.log_rotate_interval, backupCount=options.log_file_num_backups, encoding='utf-8')
        else:
            error_message = 'The value of log_rotate_mode option should be ' + '"size" or "time", not "%s".' % rotate_mode
            raise ValueError(error_message)
        channel.setFormatter(LogFormatter(color=False))
        logger.addHandler(channel)
    if options.log_to_stderr or (options.log_to_stderr is None and (not logger.handlers)):
        channel = logging.StreamHandler()
        channel.setFormatter(LogFormatter())
        logger.addHandler(channel)