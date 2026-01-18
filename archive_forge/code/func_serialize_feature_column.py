import six
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.ops import init_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@doc_controls.header(_FEATURE_COLUMN_DEPRECATION_WARNING)
@tf_export('__internal__.feature_column.serialize_feature_column', v1=[])
@deprecation.deprecated(None, _FEATURE_COLUMN_DEPRECATION_RUNTIME_WARNING)
def serialize_feature_column(fc):
    """Serializes a FeatureColumn or a raw string key.

  This method should only be used to serialize parent FeatureColumns when
  implementing FeatureColumn.get_config(), else serialize_feature_columns()
  is preferable.

  This serialization also keeps information of the FeatureColumn class, so
  deserialization is possible without knowing the class type. For example:

  a = numeric_column('x')
  a.get_config() gives:
  {
      'key': 'price',
      'shape': (1,),
      'default_value': None,
      'dtype': 'float32',
      'normalizer_fn': None
  }
  While serialize_feature_column(a) gives:
  {
      'class_name': 'NumericColumn',
      'config': {
          'key': 'price',
          'shape': (1,),
          'default_value': None,
          'dtype': 'float32',
          'normalizer_fn': None
      }
  }

  Args:
    fc: A FeatureColumn or raw feature key string.

  Returns:
    Keras serialization for FeatureColumns, leaves string keys unaffected.

  Raises:
    ValueError if called with input that is not string or FeatureColumn.
  """
    if isinstance(fc, six.string_types):
        return fc
    elif isinstance(fc, fc_types.FeatureColumn):
        return {'class_name': fc.__class__.__name__, 'config': fc.get_config()}
    else:
        raise ValueError('Instance: {} is not a FeatureColumn'.format(fc))