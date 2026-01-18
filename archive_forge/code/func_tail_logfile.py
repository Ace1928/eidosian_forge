import logging
import os
import time
from concurrent.futures._base import Future
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Event
from typing import Dict, List, Optional, TextIO
def tail_logfile(header: str, file: str, dst: TextIO, finished: Event, interval_sec: float):
    while not os.path.exists(file):
        if finished.is_set():
            return
        time.sleep(interval_sec)
    with open(file) as fp:
        while True:
            line = fp.readline()
            if line:
                dst.write(f'{header}{line}')
            elif finished.is_set():
                break
            else:
                time.sleep(interval_sec)