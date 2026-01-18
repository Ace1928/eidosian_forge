from numba import types
from numba.cuda.stubs import _vector_type_stubs
def make_simulated_vector_type(num_elements, name):
    obj = type(name, (SimulatedVectorType,), {'num_elements': num_elements, 'base_type': types.float32, 'name': name})
    obj.user_facing_object = obj
    return obj