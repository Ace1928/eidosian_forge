import six
from tensorflow.python.feature_column import feature_column_v2_types as fc_types
from tensorflow.python.ops import init_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def register_feature_column(fc):
    """Decorator that registers a FeatureColumn for serialization."""
    _FEATURE_COLUMNS.append(fc)
    return fc