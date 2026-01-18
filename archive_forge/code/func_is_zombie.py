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
def is_zombie(self):
    return self._job_queue_state == 'zombie' or self._job_queue_state == 'finished'