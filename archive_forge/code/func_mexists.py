from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def mexists(self, key, *items):
    """
        Check whether an `items` exist in Cuckoo Filter `key`.
        For more information see `CF.MEXISTS <https://redis.io/commands/cf.mexists>`_.
        """
    return self.execute_command(CF_MEXISTS, key, *items)