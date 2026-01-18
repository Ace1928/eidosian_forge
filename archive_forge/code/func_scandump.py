from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def scandump(self, key, iter):
    """
        Begin an incremental save of the Cuckoo filter `key`.

        This is useful for large Cuckoo filters which cannot fit into the normal
        SAVE and RESTORE model.
        The first time this command is called, the value of `iter` should be 0.
        This command will return successive (iter, data) pairs until
        (0, NULL) to indicate completion.
        For more information see `CF.SCANDUMP <https://redis.io/commands/cf.scandump>`_.
        """
    return self.execute_command(CF_SCANDUMP, key, iter)