import os
import sys
import logging
def getproctitle() -> str:
    logger.debug('setproctitle C module not available')
    return ' '.join(sys.argv)