from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def reserve(self, key, k, width, depth, decay):
    """
        Create a new Top-K Filter `key` with desired probability of false
        positives `errorRate` expected entries to be inserted as `size`.
        For more information see `TOPK.RESERVE <https://redis.io/commands/topk.reserve>`_.
        """
    return self.execute_command(TOPK_RESERVE, key, k, width, depth, decay)