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
def print_dictionary(self):
    """For debugging"""
    for vv in list(self._task_dictionary.values()):
        sge_debug_print(str(vv))