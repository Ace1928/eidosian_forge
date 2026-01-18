import logging
import logging.handlers
import sys
from tornado.escape import _unicode
from tornado.util import unicode_type, basestring_type
from typing import Dict, Any, cast, Optional
def define_logging_options(options: Any=None) -> None:
    """Add logging-related flags to ``options``.

    These options are present automatically on the default options instance;
    this method is only necessary if you have created your own `.OptionParser`.

    .. versionadded:: 4.2
        This function existed in prior versions but was broken and undocumented until 4.2.
    """
    if options is None:
        import tornado.options
        options = tornado.options.options
    options.define('logging', default='info', help="Set the Python log level. If 'none', tornado won't touch the logging configuration.", metavar='debug|info|warning|error|none')
    options.define('log_to_stderr', type=bool, default=None, help='Send log output to stderr (colorized if possible). By default use stderr if --log_file_prefix is not set and no other logging is configured.')
    options.define('log_file_prefix', type=str, default=None, metavar='PATH', help='Path prefix for log files. Note that if you are running multiple tornado processes, log_file_prefix must be different for each of them (e.g. include the port number)')
    options.define('log_file_max_size', type=int, default=100 * 1000 * 1000, help='max size of log files before rollover')
    options.define('log_file_num_backups', type=int, default=10, help='number of log files to keep')
    options.define('log_rotate_when', type=str, default='midnight', help="specify the type of TimedRotatingFileHandler interval other options:('S', 'M', 'H', 'D', 'W0'-'W6')")
    options.define('log_rotate_interval', type=int, default=1, help='The interval value of timed rotating')
    options.define('log_rotate_mode', type=str, default='size', help='The mode of rotating files(time or size)')
    options.add_parse_callback(lambda: enable_pretty_logging(options))