from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
@property
def occurs(self) -> Tuple[int, Optional[int]]:
    return (self.min_occurs, self.max_occurs)