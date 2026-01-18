from contextlib import contextmanager
from operator import length_hint
from os import cpu_count
from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning
def process_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ProcessPoolExecutor.__init__`.
        [default: min(32, cpu_count() + 4)].
    chunksize  : int, optional
        Size of chunks sent to worker processes; passed to
        `concurrent.futures.ProcessPoolExecutor.map`. [default: 1].
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: mp_lock].
    """
    from concurrent.futures import ProcessPoolExecutor
    if iterables and 'chunksize' not in tqdm_kwargs:
        longest_iterable_len = max(map(length_hint, iterables))
        if longest_iterable_len > 1000:
            from warnings import warn
            warn('Iterable length %d > 1000 but `chunksize` is not set. This may seriously degrade multiprocess performance. Set `chunksize=1` or more.' % longest_iterable_len, TqdmWarning, stacklevel=2)
    if 'lock_name' not in tqdm_kwargs:
        tqdm_kwargs = tqdm_kwargs.copy()
        tqdm_kwargs['lock_name'] = 'mp_lock'
    return _executor_map(ProcessPoolExecutor, fn, *iterables, **tqdm_kwargs)