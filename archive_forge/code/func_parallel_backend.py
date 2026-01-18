import contextlib
from multiprocessing import Pool, RLock
from tqdm.auto import tqdm
from ..utils import experimental, logging
@experimental
@contextlib.contextmanager
def parallel_backend(backend_name: str):
    """
    **Experimental.**  Configures the parallel backend for parallelized dataset loading, which uses the parallelization
    implemented by joblib.

    Args:
        backend_name (str): Name of backend for parallelization implementation, has to be supported by joblib.

     Example usage:
     ```py
     with parallel_backend('spark'):
       dataset = load_dataset(..., num_proc=2)
     ```
    """
    ParallelBackendConfig.backend_name = backend_name
    if backend_name == 'spark':
        from joblibspark import register_spark
        register_spark()
    try:
        yield
    finally:
        ParallelBackendConfig.backend_name = None