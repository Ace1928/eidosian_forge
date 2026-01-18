from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def initbyprob(self, key, error, probability):
    """
        Initialize a Count-Min Sketch `key` to characteristics (`error`, `probability`) specified by user.
        For more information see `CMS.INITBYPROB <https://redis.io/commands/cms.initbyprob>`_.
        """
    return self.execute_command(CMS_INITBYPROB, key, error, probability)