import logging
import os
from typing import List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
import subprocess
def modify_context(self, uris: List[str], runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: Optional[logging.Logger]=default_logger) -> None:
    mpi_config = runtime_env.mpi()
    if mpi_config is None:
        return
    try:
        proc = subprocess.run(['mpirun', '--version'], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        logger.exception('Failed to run mpi run. Please make sure mpi has been installed')
        raise
    logger.info(f'Running MPI plugin\n {proc.stdout.decode()}')
    worker_entry = mpi_config.get('worker_entry')
    assert worker_entry is not None, '`worker_entry` must be setup in the runtime env.'
    cmds = ['mpirun'] + mpi_config.get('args', []) + [context.py_executable, '-m', 'ray._private.runtime_env.mpi_runner', worker_entry]
    context.py_executable = ' '.join(cmds)