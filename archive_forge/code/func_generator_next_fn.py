import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import script_ops
def generator_next_fn(iterator_id_t):
    """Generates the next element from iterator with ID `iterator_id_t`.

    We map this function across an infinite repetition of the
    `iterator_id_t`, and raise `StopIteration` to terminate the iteration.

    Args:
      iterator_id_t: A `tf.int64` tensor whose value uniquely identifies the
        iterator in `generator_state` from which to generate an element.

    Returns:
      The next element to generate from the iterator.
    """
    if output_types and output_shapes:
        flattened_types = [dtypes.as_dtype(dt) for dt in nest.flatten(output_types)]
        flattened_shapes = nest.flatten(output_shapes)

        def generator_py_func(iterator_id):
            """A `py_func` that will be called to invoke the iterator."""
            values = next(generator_state.get_iterator(iterator_id))
            try:
                flattened_values = nest.flatten_up_to(output_types, values)
            except (TypeError, ValueError) as e:
                raise TypeError(f'`generator` yielded an element that did not match the expected structure. The expected structure was {output_types}, but the yielded element was {values}.') from e
            ret_arrays = []
            for ret, dtype in zip(flattened_values, flattened_types):
                try:
                    ret_arrays.append(script_ops.FuncRegistry._convert(ret, dtype=dtype.as_numpy_dtype))
                except (TypeError, ValueError) as e:
                    raise TypeError(f'`generator` yielded an element that could not be converted to the expected type. The expected type was {dtype.name}, but the yielded element was {ret}.') from e
            for ret_array, expected_dtype, expected_shape in zip(ret_arrays, flattened_types, flattened_shapes):
                if ret_array.dtype != expected_dtype.as_numpy_dtype:
                    raise TypeError(f'`generator` yielded an element of type {ret_array.dtype} where an element of type {expected_dtype.as_numpy_dtype} was expected.')
                if not expected_shape.is_compatible_with(ret_array.shape):
                    raise TypeError(f'`generator` yielded an element of shape {ret_array.shape} where an element of shape {expected_shape} was expected.')
            return ret_arrays
        flat_values = script_ops.numpy_function(generator_py_func, [iterator_id_t], flattened_types)
        if not isinstance(flat_values, (list, tuple)):
            flat_values = [flat_values]
        if output_shapes is not None:
            for ret_t, shape in zip(flat_values, flattened_shapes):
                ret_t.set_shape(shape)
        return nest.pack_sequence_as(output_types, flat_values)
    else:
        flat_output_types = structure.get_flat_tensor_types(output_signature)

        def generator_py_func(iterator_id):
            """A `py_func` that will be called to invoke the iterator."""
            values = next(generator_state.get_iterator(iterator_id.numpy()))
            try:
                values = structure.normalize_element(values, output_signature)
            except (TypeError, ValueError) as e:
                raise TypeError(f'`generator` yielded an element that did not match the expected structure. The expected structure was {output_signature}, but the yielded element was {values}.') from e
            values_spec = structure.type_spec_from_value(values)
            if not structure.are_compatible(values_spec, output_signature):
                raise TypeError(f'`generator` yielded an element of {values_spec} where an element of {output_signature} was expected.')
            return structure.to_tensor_list(output_signature, values)
        return script_ops.eager_py_func(generator_py_func, inp=[iterator_id_t], Tout=flat_output_types)