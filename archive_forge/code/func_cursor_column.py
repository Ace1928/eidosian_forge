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
def cursor_column(self, column):
    self.cursor.x = min(column, 1) - 1