from typing import Callable, Tuple, Any
def jax_unflatten(aux, parameters):
    return unflatten_fn(parameters, aux)