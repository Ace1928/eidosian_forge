import asyncio
import logging
import os
import signal
import sys
import traceback
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Iterable, Optional, Set
from mypy_extensions import mypyc_attr
from black import WriteBack, format_file_in_place
from black.cache import Cache
from black.mode import Mode
from black.output import err
from black.report import Changed, Report
@mypyc_attr(patchable=True)
def reformat_many(sources: Set[Path], fast: bool, write_back: WriteBack, mode: Mode, report: Report, workers: Optional[int]) -> None:
    """Reformat multiple files using a ProcessPoolExecutor."""
    maybe_install_uvloop()
    executor: Executor
    if workers is None:
        workers = int(os.environ.get('BLACK_NUM_WORKERS', 0))
        workers = workers or os.cpu_count() or 1
    if sys.platform == 'win32':
        workers = min(workers, 60)
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
    except (ImportError, NotImplementedError, OSError):
        executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(schedule_formatting(sources=sources, fast=fast, write_back=write_back, mode=mode, report=report, loop=loop, executor=executor))
    finally:
        try:
            shutdown(loop)
        finally:
            asyncio.set_event_loop(None)
        if executor is not None:
            executor.shutdown()