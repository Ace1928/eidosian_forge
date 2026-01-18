import six
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.ops import init_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('__internal__.feature_column.deserialize_feature_column', v1=[])
def deserialize_feature_column(config, custom_objects=None, columns_by_name=None):
    """Deserializes a `config` generated with `serialize_feature_column`.

  This method should only be used to deserialize parent FeatureColumns when
  implementing FeatureColumn.from_config(), else deserialize_feature_columns()
  is preferable. Returns a FeatureColumn for this config.

  Args:
    config: A Dict with the serialization of feature columns acquired by
      `serialize_feature_column`, or a string representing a raw column.
    custom_objects: A Dict from custom_object name to the associated keras
      serializable objects (FeatureColumns, classes or functions).
    columns_by_name: A Dict[String, FeatureColumn] of existing columns in order
      to avoid duplication.

  Raises:
    ValueError if `config` has invalid format (e.g: expected keys missing,
    or refers to unknown classes).

  Returns:
    A FeatureColumn corresponding to the input `config`.
  """
    if isinstance(config, six.string_types):
        return config
    module_feature_column_classes = {cls.__name__: cls for cls in _FEATURE_COLUMNS}
    if columns_by_name is None:
        columns_by_name = {}
    cls, cls_config = _class_and_config_for_serialized_keras_object(config, module_objects=module_feature_column_classes, custom_objects=custom_objects, printable_module_name='feature_column_v2')
    if not issubclass(cls, fc_types.FeatureColumn):
        raise ValueError('Expected FeatureColumn class, instead found: {}'.format(cls))
    new_instance = cls.from_config(cls_config, custom_objects=custom_objects, columns_by_name=columns_by_name)
    return columns_by_name.setdefault(_column_name_with_class_name(new_instance), new_instance)