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
def erase_screen(self, mode=0):
    if mode == 0:
        for i in range(self.cursor.y + 1, self.num_lines):
            if i in self.buffer:
                del self.buffer[i]
        self.erase_line(mode)
    if mode == 1:
        for i in range(self.cursor.y):
            if i in self.buffer:
                del self.buffer[i]
        self.erase_line(mode)
    elif mode == 2 or mode == 3:
        self.buffer.clear()