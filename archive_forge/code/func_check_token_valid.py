from pyomo.common import Factory
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import MouseTrap
from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TransformationTimer
def check_token_valid(self, cls, model, targets):
    if cls is not self._transformation:
        raise ValueError("Attempting to reverse transformation of class '%s' using a token created by a transformation of class '%s'. Cannot revert transformation with a token from another transformation." % (cls, self._transformation))
    if model is not self._model:
        raise MouseTrap("A reverse transformation was called on model '%s', but the transformation that created this token was created from model '%s'. We do not currently support reversing transformations on clones of the transformed model." % (model.name, self._model.name))