import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
def sge_debug_print(message):
    """Needed for debugging on big jobs.  Once this is fully vetted, it can be removed."""
    logger.debug(DEBUGGING_PREFIX + ' ' + '=!' * 3 + '  ' + message)