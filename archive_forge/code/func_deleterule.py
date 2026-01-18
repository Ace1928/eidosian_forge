from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
def deleterule(self, source_key: KeyT, dest_key: KeyT):
    """
        Delete a compaction rule from `source_key` to `dest_key`..

        For more information: https://redis.io/commands/ts.deleterule/
        """
    return self.execute_command(DELETERULE_CMD, source_key, dest_key)