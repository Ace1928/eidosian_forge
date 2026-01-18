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
def erase_line(self, mode=0):
    curr_line = self.buffer[self.cursor.y]
    if mode == 0:
        for i in range(self.cursor.x, self._get_line_len(self.cursor.y)):
            if i in curr_line:
                del curr_line[i]
    elif mode == 1:
        for i in range(self.cursor.x + 1):
            if i in curr_line:
                del curr_line[i]
    else:
        curr_line.clear()