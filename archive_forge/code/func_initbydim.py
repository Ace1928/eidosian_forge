from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def initbydim(self, key, width, depth):
    """
        Initialize a Count-Min Sketch `key` to dimensions (`width`, `depth`) specified by user.
        For more information see `CMS.INITBYDIM <https://redis.io/commands/cms.initbydim>`_.
        """
    return self.execute_command(CMS_INITBYDIM, key, width, depth)