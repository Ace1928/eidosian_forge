import logging
import sys
import os
def init_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv('LOG_LEVEL', 'DEBUG'))
    if VLLM_CONFIGURE_LOGGING:
        logger.addHandler(_default_handler)
        logger.propagate = False
    return logger