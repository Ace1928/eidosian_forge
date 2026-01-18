from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def loadchunk(self, key, iter, data):
    """
        Restore a filter previously saved using SCANDUMP. See the SCANDUMP command for example usage.

        This command will overwrite any Cuckoo filter stored under key.
        Ensure that the Cuckoo filter will not be modified between invocations.
        For more information see `CF.LOADCHUNK <https://redis.io/commands/cf.loadchunk>`_.
        """
    return self.execute_command(CF_LOADCHUNK, key, iter, data)