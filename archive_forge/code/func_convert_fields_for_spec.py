import collections
import collections.abc
import enum
import typing
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.util import type_annotations
def convert_fields_for_spec(fields, field_values):
    """Type-checks and converts field values for a TypeSpec (in place).

  This is similar to `convert_fields`, except that we expect a `TypeSpec` for
  tensor-like types.  In particular, if the `value_type` of a field is
  `tf.Tensor` or a `CompositeTensor` subclass, then the corresponding value in
  `fields` is expected to contain a `TypeSpec` (rather than a value described by
  that `TypeSpec`).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
    _convert_fields(fields, field_values, context=_ConversionContext.SPEC)