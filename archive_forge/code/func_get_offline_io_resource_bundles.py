from typing import Dict, List, TYPE_CHECKING
from ray.rllib.utils.annotations import PublicAPI
@PublicAPI
def get_offline_io_resource_bundles(config: 'AlgorithmConfig') -> List[Dict[str, float]]:
    if config.input_ == 'dataset':
        input_config = config.input_config
        parallelism = input_config.get('parallelism', config.get('num_workers', 1))
        cpus_per_task = input_config.get('num_cpus_per_read_task', DEFAULT_NUM_CPUS_PER_TASK)
        return [{'CPU': cpus_per_task} for _ in range(parallelism)]
    else:
        return []