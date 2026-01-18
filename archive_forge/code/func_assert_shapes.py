import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['debugging.assert_shapes'])
@dispatch.add_dispatch_support
def assert_shapes(shapes, data=None, summarize=None, message=None, name=None):
    """Assert tensor shapes and dimension size relationships between tensors.

  This Op checks that a collection of tensors shape relationships
  satisfies given constraints.

  Example:

  >>> n = 10
  >>> q = 3
  >>> d = 7
  >>> x = tf.zeros([n,q])
  >>> y = tf.ones([n,d])
  >>> param = tf.Variable([1.0, 2.0, 3.0])
  >>> scalar = 1.0
  >>> tf.debugging.assert_shapes([
  ...  (x, ('N', 'Q')),
  ...  (y, ('N', 'D')),
  ...  (param, ('Q',)),
  ...  (scalar, ()),
  ... ])

  >>> tf.debugging.assert_shapes([
  ...   (x, ('N', 'D')),
  ...   (y, ('N', 'D'))
  ... ])
  Traceback (most recent call last):
  ...
  ValueError: ...

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_shapes(shapes)]):
    output = tf.matmul(x, y, transpose_a=True)
  ```

  If `x`, `y`, `param` or `scalar` does not have a shape that satisfies
  all specified constraints, `message`, as well as the first `summarize` entries
  of the first encountered violating tensor are printed, and
  `InvalidArgumentError` is raised.

  Size entries in the specified shapes are checked against other entries by
  their __hash__, except:
    - a size entry is interpreted as an explicit size if it can be parsed as an
      integer primitive.
    - a size entry is interpreted as *any* size if it is None or '.'.

  If the first entry of a shape is `...` (type `Ellipsis`) or '*' that indicates
  a variable number of outer dimensions of unspecified size, i.e. the constraint
  applies to the inner-most dimensions only.

  Scalar tensors and specified shapes of length zero (excluding the 'inner-most'
  prefix) are both treated as having a single dimension of size one.

  Args:
    shapes: A list of (`Tensor`, `shape`) tuples, wherein `shape` is the
      expected shape of `Tensor`. See the example code above. The `shape` must
      be an iterable. Each element of the iterable can be either a concrete
      integer value or a string that abstractly represents the dimension.
      For example,
        - `('N', 'Q')` specifies a 2D shape wherein the first and second
          dimensions of shape may or may not be equal.
        - `('N', 'N', 'Q')` specifies a 3D shape wherein the first and second
          dimensions are equal.
        - `(1, 'N')` specifies a 2D shape wherein the first dimension is
          exactly 1 and the second dimension can be any value.
      Note that the abstract dimension letters take effect across different
      tuple elements of the list. For example,
      `tf.debugging.assert_shapes([(x, ('N', 'A')), (y, ('N', 'B'))]` asserts
      that both `x` and `y` are rank-2 tensors and their first dimensions are
      equal (`N`).
      `shape` can also be a `tf.TensorShape`.
    data: The tensors to print out if the condition is False.  Defaults to error
      message and first few entries of the violating tensor.
    summarize: Print this many entries of the tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_shapes".

  Returns:
    Op raising `InvalidArgumentError` unless all shape constraints are
    satisfied.
    If static checks determine all constraints are satisfied, a `no_op` is
    returned.

  Raises:
    ValueError:  If static checks determine any shape constraint is violated.
  """
    if isinstance(shapes, dict):
        shapes = shapes.items()
    message_prefix = _message_prefix(message)
    with ops.name_scope(name, 'assert_shapes', [shapes, data]):
        shape_constraints = [(x if isinstance(x, sparse_tensor.SparseTensor) else ops.convert_to_tensor(x), s) for x, s in shapes if s is not None]
        executing_eagerly = context.executing_eagerly()

        def tensor_name(x):
            if executing_eagerly or isinstance(x, sparse_tensor.SparseTensor):
                return _shape_and_dtype_str(x)
            return x.name
        tensor_dim_sizes = []
        for tensor, symbolic_shape in shape_constraints:
            is_iterable = hasattr(symbolic_shape, '__iter__') or hasattr(symbolic_shape, '__getitem__')
            if not is_iterable:
                raise ValueError('%sTensor %s.  Specified shape must be an iterable.  An iterable has the attribute `__iter__` or `__getitem__`.  Received specified shape: %s' % (message_prefix, tensor_name(tensor), symbolic_shape))
            symbolic_shape_tuple = tuple(symbolic_shape)
            tensors_specified_innermost = False
            for i, symbol in enumerate(symbolic_shape_tuple):
                if symbol not in [Ellipsis, '*']:
                    continue
                if i != 0:
                    raise ValueError('%sTensor %s specified shape index %d.  Symbol `...` or `*` for a variable number of unspecified dimensions is only allowed as the first entry' % (message_prefix, tensor_name(tensor), i))
                tensors_specified_innermost = True
            tensor_dim_sizes.append(_TensorDimSizes(tensor, tensors_specified_innermost, _dimension_sizes(tensor), _symbolic_dimension_sizes(symbolic_shape_tuple[1:] if tensors_specified_innermost else symbolic_shape_tuple)))
        rank_assertions = []
        for sizes in tensor_dim_sizes:
            rank = len(sizes.symbolic_sizes)
            rank_zero_or_one = rank in [0, 1]
            if sizes.unspecified_dim:
                if rank_zero_or_one:
                    continue
                assertion = assert_rank_at_least(x=sizes.x, rank=rank, data=data, summarize=summarize, message=message, name=name)
            elif rank_zero_or_one:
                assertion = assert_rank_in(x=sizes.x, ranks=[0, 1], data=data, summarize=summarize, message=message, name=name)
            else:
                assertion = assert_rank(x=sizes.x, rank=rank, data=data, summarize=summarize, message=message, name=name)
            rank_assertions.append(assertion)
        size_assertions = []
        size_specifications = {}
        for sizes in tensor_dim_sizes:
            for i, size_symbol in enumerate(sizes.symbolic_sizes):
                if _is_symbol_for_any_size(size_symbol):
                    continue
                if sizes.unspecified_dim:
                    tensor_dim = i - len(sizes.symbolic_sizes)
                else:
                    tensor_dim = i
                if size_symbol in size_specifications or _has_known_value(size_symbol):
                    if _has_known_value(size_symbol):
                        specified_size = int(size_symbol)
                        size_check_message = 'Specified explicitly'
                    else:
                        specified_size, specified_by_y, specified_at_dim = size_specifications[size_symbol]
                        size_check_message = 'Specified by tensor %s dimension %d' % (tensor_name(specified_by_y), specified_at_dim)
                    with ops.control_dependencies(rank_assertions):
                        actual_size = sizes.actual_sizes[tensor_dim]
                    if _has_known_value(actual_size) and _has_known_value(specified_size):
                        if int(actual_size) != int(specified_size):
                            raise ValueError('%s%s.  Tensor %s dimension %s must have size %d.  Received size %d, shape %s' % (message_prefix, size_check_message, tensor_name(sizes.x), tensor_dim, specified_size, actual_size, sizes.x.get_shape()))
                        continue
                    condition = math_ops.equal(ops.convert_to_tensor(actual_size), ops.convert_to_tensor(specified_size))
                    data_ = data
                    if data is None:
                        data_ = [message_prefix, size_check_message, 'Tensor %s dimension' % tensor_name(sizes.x), tensor_dim, 'must have size', specified_size, 'Received shape: ', array_ops.shape(sizes.x)]
                    size_assertions.append(control_flow_assert.Assert(condition, data_, summarize=summarize))
                else:
                    with ops.control_dependencies(rank_assertions):
                        size = sizes.actual_sizes[tensor_dim]
                    size_specifications[size_symbol] = (size, sizes.x, tensor_dim)
    with ops.control_dependencies(rank_assertions):
        shapes_assertion = control_flow_ops.group(size_assertions)
    return shapes_assertion