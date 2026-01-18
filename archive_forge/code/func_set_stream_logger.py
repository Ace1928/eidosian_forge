import logging
from boto3.session import Session
def set_stream_logger(name='boto3', level=logging.DEBUG, format_string=None):
    """
    Add a stream handler for the given name and level to the logging module.
    By default, this logs all boto3 messages to ``stdout``.

        >>> import boto3
        >>> boto3.set_stream_logger('boto3.resources', logging.INFO)

    For debugging purposes a good choice is to set the stream logger to ``''``
    which is equivalent to saying "log everything".

    .. WARNING::
       Be aware that when logging anything from ``'botocore'`` the full wire
       trace will appear in your logs. If your payloads contain sensitive data
       this should not be used in production.

    :type name: string
    :param name: Log name
    :type level: int
    :param level: Logging level, e.g. ``logging.INFO``
    :type format_string: str
    :param format_string: Log message format
    """
    if format_string is None:
        format_string = '%(asctime)s %(name)s [%(levelname)s] %(message)s'
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)