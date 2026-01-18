from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
def is_emptiable(self) -> bool:
    """
        Tests if max_occurs == 0. A zero-length model group is considered emptiable.
        For model groups the test outcome depends also on nested particles.
        """
    return self.min_occurs == 0