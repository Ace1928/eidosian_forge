import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
def configure_component_logger(*, component_name: str, component_id: str, logging_config: LoggingConfig, component_type: Optional[ServeComponentType]=None):
    """Configure a logger to be used by a Serve component.

    The logger will log using a standard format to make components identifiable
    using the provided name and unique ID for this instance (e.g., replica ID).

    This logger will *not* propagate its log messages to the parent logger(s).
    """
    logger = logging.getLogger(SERVE_LOGGER_NAME)
    logger.propagate = False
    logger.setLevel(logging_config.log_level)
    logger.handlers.clear()
    factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        request_context = ray.serve.context._serve_request_context.get()
        record = factory(*args, **kwargs)
        if request_context.route:
            setattr(record, SERVE_LOG_ROUTE, request_context.route)
        if request_context.request_id:
            setattr(record, SERVE_LOG_REQUEST_ID, request_context.request_id)
        if request_context.app_name:
            setattr(record, SERVE_LOG_APPLICATION, request_context.app_name)
        return record
    logging.setLogRecordFactory(record_factory)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ServeFormatter(component_name, component_id))
    stream_handler.addFilter(log_to_stderr_filter)
    logger.addHandler(stream_handler)
    if logging_config.logs_dir:
        logs_dir = logging_config.logs_dir
    else:
        logs_dir = get_serve_logs_dir()
    os.makedirs(logs_dir, exist_ok=True)
    max_bytes = ray._private.worker._global_node.max_bytes
    backup_count = ray._private.worker._global_node.backup_count
    log_file_name = get_component_log_file_name(component_name=component_name, component_id=component_id, component_type=component_type, suffix='.log')
    file_handler = logging.handlers.RotatingFileHandler(os.path.join(logs_dir, log_file_name), maxBytes=max_bytes, backupCount=backup_count)
    if RAY_SERVE_ENABLE_JSON_LOGGING:
        logger.warning("'RAY_SERVE_ENABLE_JSON_LOGGING' is deprecated, please use 'LoggingConfig' to enable json format.")
    if RAY_SERVE_ENABLE_JSON_LOGGING or logging_config.encoding == EncodingType.JSON:
        file_handler.setFormatter(ServeJSONFormatter(component_name, component_id, component_type))
    else:
        file_handler.setFormatter(ServeFormatter(component_name, component_id))
    if logging_config.enable_access_log is False:
        file_handler.addFilter(log_access_log_filter)
    logger.addHandler(file_handler)