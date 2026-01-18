from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
def is_single(self) -> bool:
    """
        Tests if the particle has max_occurs == 1. For elements the test
        outcome depends also on parent group. For model groups the test
        outcome depends also on nested model groups.
        """
    return self.max_occurs == 1