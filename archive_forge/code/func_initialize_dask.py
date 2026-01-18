import os
from modin.config import (
from modin.core.execution.utils import set_env
def initialize_dask():
    """Initialize Dask environment."""
    from distributed.client import default_client
    try:
        client = default_client()

        def _disable_warnings():
            import warnings
            warnings.simplefilter('ignore', category=FutureWarning)
        client.run(_disable_warnings)
    except ValueError:
        from distributed import Client
        num_cpus = CpuCount.get()
        threads_per_worker = DaskThreadsPerWorker.get()
        memory_limit = Memory.get()
        worker_memory_limit = memory_limit // num_cpus if memory_limit else 'auto'
        with set_env(PYTHONWARNINGS='ignore::FutureWarning'):
            client = Client(n_workers=num_cpus, threads_per_worker=threads_per_worker, memory_limit=worker_memory_limit)
        if GithubCI.get():
            access_key = CIAWSAccessKeyID.get()
            aws_secret = CIAWSSecretAccessKey.get()
            client.run(lambda: os.environ.update({'AWS_ACCESS_KEY_ID': access_key, 'AWS_SECRET_ACCESS_KEY': aws_secret}))
    num_cpus = len(client.ncores())
    NPartitions._put(num_cpus)