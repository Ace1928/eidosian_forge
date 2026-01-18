import asyncio
import atexit
import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener
from typing import Dict, List, Optional, Tuple, Union
def setup_logging_queues() -> None:
    if sys.version_info.major < 3:
        raise RuntimeError('This feature requires Python 3.')
    queue_listeners: List[AsyncQueueListener] = []
    previous_queue_listeners = []
    for logger_name in get_all_logger_names(include_root=True):
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            ori_handlers: List[logging.Handler] = []
            if logger_name in GLOBAL_LOGGER_HANDLERS:
                ori_handlers.extend(GLOBAL_LOGGER_HANDLERS[logger_name][0])
                for handler in ori_handlers:
                    handler.createLock()
                logger.handlers = []
                logger.handlers.extend(ori_handlers)
                previous_queue_listeners.append(GLOBAL_LOGGER_HANDLERS[logger_name][1])
            else:
                ori_handlers.extend(logger.handlers)
            log_queue: queue.Queue[str] = queue.Queue(-1)
            queue_handler = QueueHandler(log_queue)
            queue_listener = AsyncQueueListener(log_queue, respect_handler_level=True)
            queuify_logger(logger, queue_handler, queue_listener)
            queue_listeners.append(queue_listener)
            GLOBAL_LOGGER_HANDLERS[logger_name] = (ori_handlers, queue_listener)
    stop_queue_listeners(*previous_queue_listeners)
    for listener in queue_listeners:
        listener.start()
    atexit.register(stop_queue_listeners, *queue_listeners)