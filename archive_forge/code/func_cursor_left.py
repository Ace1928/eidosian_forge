import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
def cursor_left(self, n=1):
    n = min(n, self.cursor.x)
    self.cursor.x -= n