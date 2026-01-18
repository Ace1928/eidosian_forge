from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
@property
def effective_min_occurs(self) -> int:
    """
        A property calculated from minOccurs, that is equal to minOccurs
        for elements and may vary for content model groups, in dependance
        of group model and structure. Used for checking restrictions of
        model groups in XSD 1.1.
        """
    return self.min_occurs