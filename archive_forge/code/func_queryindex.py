from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
def queryindex(self, filters: List[str]):
    """# noqa
        Get all time series keys matching the `filter` list.

        For more information: https://redis.io/commands/ts.queryindex/
        """
    return self.execute_command(QUERYINDEX_CMD, *filters)