import unidist
import unidist.config as unidist_cfg
import modin.config as modin_cfg
from .engine_wrapper import UnidistWrapper
def initialize_unidist():
    """
    Initialize unidist based on ``modin.config`` variables and internal defaults.
    """
    if unidist_cfg.Backend.get() != 'mpi':
        raise RuntimeError(f"Modin only supports MPI through unidist for now, got unidist backend '{unidist_cfg.Backend.get()}'")
    if not unidist.is_initialized():
        modin_cfg.CpuCount.subscribe(lambda cpu_count: unidist_cfg.CpuCount.put(cpu_count.get()))
        unidist_cfg.MpiRuntimeEnv.put({'env_vars': {'PYTHONWARNINGS': 'ignore::FutureWarning'}})
        unidist.init()
    num_cpus = sum((v['CPU'] for v in unidist.cluster_resources().values()))
    modin_cfg.NPartitions._put(num_cpus)