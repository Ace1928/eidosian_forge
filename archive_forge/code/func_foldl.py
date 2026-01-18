from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.gen_functional_ops import remote_call
from tensorflow.python.ops.gen_functional_ops import symbolic_gradient
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['foldl'])
@dispatch.add_dispatch_support
def foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None):
    """foldl on the list of tensors unpacked from `elems` on dimension 0.

  This foldl operator repeatedly applies the callable `fn` to a sequence
  of elements from first to last. The elements are made of the tensors
  unpacked from `elems` on dimension 0. The callable fn takes two tensors as
  arguments. The first argument is the accumulated value computed from the
  preceding invocation of fn, and the second is the value at the current
  position of `elems`. If `initializer` is None, `elems` must contain at least
  one element, and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is fn(initializer, values[0]).shape`.

  This method also allows multi-arity `elems` and output of `fn`.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The signature of `fn` may
  match the structure of `elems`.  That is, if `elems` is
  `(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
  `fn = lambda (t1, [t2, t3, [t4, t5]]):`.

  Args:
    fn: The callable to be performed.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unpacked along their first dimension.  The nested sequence of the
      resulting slices will be the first argument to `fn`.
    initializer: (optional) A tensor or (possibly nested) sequence of tensors,
      as the initial value for the accumulator.
    parallel_iterations: (optional) The number of iterations allowed to run in
      parallel.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors, resulting from applying
    `fn` consecutively to the list of tensors unpacked from `elems`, from first
    to last.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = tf.constant([1, 2, 3, 4, 5, 6])
    sum = foldl(lambda a, x: a + x, elems)
    # sum == 21
    ```
  """
    if not callable(fn):
        raise TypeError(f'{fn.__name__} is not callable. Please provide a callable function.')

    def create_ta(elem):
        return tensor_array_ops.TensorArray(dtype=elem.dtype, size=n, dynamic_size=False, infer_shape=True).unstack(elem)
    in_graph_mode = not context.executing_eagerly()
    with ops.name_scope(name, 'foldl', [elems]):
        if in_graph_mode:
            varscope = vs.get_variable_scope()
            varscope_caching_device_was_none = False
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)
                varscope_caching_device_was_none = True
        elems_flat = [ops.convert_to_tensor(elem, name='elem') for elem in nest.flatten(elems)]
        n = tensor_shape.dimension_value(elems_flat[0].shape[0]) or array_ops.shape(elems_flat[0])[0]
        elems_ta = nest.map_structure(create_ta, elems)
        if initializer is None:
            a = nest.map_structure(lambda elem: elem.read(0), elems_ta)
            i = constant_op.constant(1)
        else:
            a = initializer
            i = constant_op.constant(0)

        def compute(i, a):
            elem_i = nest.map_structure(lambda elem: elem.read(i), elems_ta)
            a = fn(a, elem_i)
            return [i + 1, a]
        _, r_a = while_loop.while_loop(lambda i, a: i < n, compute, [i, a], parallel_iterations=parallel_iterations, back_prop=back_prop, swap_memory=swap_memory, maximum_iterations=n)
        if in_graph_mode and varscope_caching_device_was_none:
            varscope.set_caching_device(None)
        return r_a