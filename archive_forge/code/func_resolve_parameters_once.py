import numbers
from typing import AbstractSet, Any, cast, TYPE_CHECKING, TypeVar
from typing_extensions import Self
import sympy
from typing_extensions import Protocol
from cirq import study
from cirq._doc import doc_private
def resolve_parameters_once(val: Any, param_resolver: 'cirq.ParamResolverOrSimilarType'):
    """Performs a single parameter resolution step using the param resolver."""
    return resolve_parameters(val, param_resolver, False)