import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
def randsign():
    """Randomly choose either 1 or -1.

    Returns
    -------
    sign : int
    """
    return random.choice((-1, 1))