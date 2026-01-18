import sys
import logging
import timeit
from functools import wraps
from collections.abc import Mapping, Callable
import warnings
from logging import PercentStyle
def configLogger(**kwargs):
    """A more sophisticated logging system configuation manager.

    This is more or less the same as :py:func:`logging.basicConfig`,
    with some additional options and defaults.

    The default behaviour is to create a ``StreamHandler`` which writes to
    sys.stderr, set a formatter using the ``DEFAULT_FORMATS`` strings, and add
    the handler to the top-level library logger ("fontTools").

    A number of optional keyword arguments may be specified, which can alter
    the default behaviour.

    Args:

            logger: Specifies the logger name or a Logger instance to be
                    configured. (Defaults to "fontTools" logger). Unlike ``basicConfig``,
                    this function can be called multiple times to reconfigure a logger.
                    If the logger or any of its children already exists before the call is
                    made, they will be reset before the new configuration is applied.
            filename: Specifies that a ``FileHandler`` be created, using the
                    specified filename, rather than a ``StreamHandler``.
            filemode: Specifies the mode to open the file, if filename is
                    specified. (If filemode is unspecified, it defaults to ``a``).
            format: Use the specified format string for the handler. This
                    argument also accepts a dictionary of format strings keyed by
                    level name, to allow customising the records appearance for
                    specific levels. The special ``'*'`` key is for 'any other' level.
            datefmt: Use the specified date/time format.
            level: Set the logger level to the specified level.
            stream: Use the specified stream to initialize the StreamHandler. Note
                    that this argument is incompatible with ``filename`` - if both
                    are present, ``stream`` is ignored.
            handlers: If specified, this should be an iterable of already created
                    handlers, which will be added to the logger. Any handler in the
                    list which does not have a formatter assigned will be assigned the
                    formatter created in this function.
            filters: If specified, this should be an iterable of already created
                    filters. If the ``handlers`` do not already have filters assigned,
                    these filters will be added to them.
            propagate: All loggers have a ``propagate`` attribute which determines
                    whether to continue searching for handlers up the logging hierarchy.
                    If not provided, the "propagate" attribute will be set to ``False``.
    """
    handlers = kwargs.pop('handlers', None)
    if handlers is None:
        if 'stream' in kwargs and 'filename' in kwargs:
            raise ValueError("'stream' and 'filename' should not be specified together")
    elif 'stream' in kwargs or 'filename' in kwargs:
        raise ValueError("'stream' or 'filename' should not be specified together with 'handlers'")
    if handlers is None:
        filename = kwargs.pop('filename', None)
        mode = kwargs.pop('filemode', 'a')
        if filename:
            h = logging.FileHandler(filename, mode)
        else:
            stream = kwargs.pop('stream', None)
            h = logging.StreamHandler(stream)
        handlers = [h]
    logger = kwargs.pop('logger', 'fontTools')
    if not logger or isinstance(logger, str):
        logger = logging.getLogger(logger)
    _resetExistingLoggers(parent=logger.name)
    fs = kwargs.pop('format', None)
    dfs = kwargs.pop('datefmt', None)
    style = kwargs.pop('style', '%')
    fmt = LevelFormatter(fs, dfs, style)
    filters = kwargs.pop('filters', [])
    for h in handlers:
        if h.formatter is None:
            h.setFormatter(fmt)
        if not h.filters:
            for f in filters:
                h.addFilter(f)
        logger.addHandler(h)
    if logger.name != 'root':
        logger.propagate = kwargs.pop('propagate', False)
    level = kwargs.pop('level', None)
    if level is not None:
        logger.setLevel(level)
    if kwargs:
        keys = ', '.join(kwargs.keys())
        raise ValueError('Unrecognised argument(s): %s' % keys)