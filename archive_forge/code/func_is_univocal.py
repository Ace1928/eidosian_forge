from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
def is_univocal(self) -> bool:
    """Tests if min_occurs == max_occurs."""
    return self.min_occurs == self.max_occurs