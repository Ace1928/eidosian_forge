from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
def has_occurs_restriction(self, other: Union[ModelParticleType, 'OccursCalculator']) -> bool:
    if self.min_occurs < other.min_occurs:
        return False
    elif self.max_occurs == 0:
        return True
    elif other.max_occurs is None:
        return True
    elif self.max_occurs is None:
        return False
    else:
        return self.max_occurs <= other.max_occurs