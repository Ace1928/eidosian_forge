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
def carriage_return(self):
    self.cursor.x = 0