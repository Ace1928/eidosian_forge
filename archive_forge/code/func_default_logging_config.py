import logging
import sys
def default_logging_config():
    """Set up the default Dulwich loggers."""
    remove_null_handler()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s %(levelname)s: %(message)s')