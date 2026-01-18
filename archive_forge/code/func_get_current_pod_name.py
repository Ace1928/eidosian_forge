from typing import Optional
from ray._private.accelerators import TPUAcceleratorManager
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
def get_current_pod_name() -> Optional[str]:
    """Return the name of the TPU pod that the worker is a part of.
    Returns:
      str: the name of the TPU pod. Returns None if not part of a TPU pod.
    """
    tpu_name = TPUAcceleratorManager.get_current_node_tpu_name()
    if tpu_name == '':
        tpu_name = None
    return tpu_name