import multiprocessing.dummy
import time
from typing import Callable
from wandb.errors.term import termlog
def notify_downloaded(self) -> None:
    with self._lock:
        self._n_files_downloaded += 1
        if self._n_files_downloaded == self._nfiles:
            self._termlog(f'  {self._nfiles} of {self._nfiles} files downloaded.  ', newline=True)
            self._last_log_time = self._clock()
        elif self._clock() - self._last_log_time > 0.1:
            self._spinner_index += 1
            spinner = '-\\|/'[self._spinner_index % 4]
            self._termlog(f'{spinner} {self._n_files_downloaded} of {self._nfiles} files downloaded...\r', newline=False)
            self._last_log_time = self._clock()