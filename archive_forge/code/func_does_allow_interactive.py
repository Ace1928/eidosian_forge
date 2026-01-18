import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
def does_allow_interactive():
    return _allow_interactive