import ray
import ray._private.profiling as profiling
import ray._private.services as services
import ray._private.utils as utils
import ray._private.worker
from ray._private import ray_constants
from ray._private.state import GlobalState
from ray._raylet import GcsClientOptions
def get_state_from_address(address=None):
    address = services.canonicalize_bootstrap_address_or_die(address)
    state = GlobalState()
    options = GcsClientOptions.from_gcs_address(address)
    state._initialize_global_state(options)
    return state