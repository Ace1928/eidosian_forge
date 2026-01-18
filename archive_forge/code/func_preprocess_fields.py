from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
def preprocess_fields(self, fields):
    """Subclasses can override this to do additional clean up on fields.

        The default is an identity function.

        Parameters:
        -----------
        fields : Sequence[Tuple[str, Type]]
        """
    return fields