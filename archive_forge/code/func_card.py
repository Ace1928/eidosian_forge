from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def card(self, key):
    """
        Returns the cardinality of a Bloom filter - number of items that were added to a Bloom filter and detected as unique
        (items that caused at least one bit to be set in at least one sub-filter).
        For more information see `BF.CARD <https://redis.io/commands/bf.card>`_.
        """
    return self.execute_command(BF_CARD, key)