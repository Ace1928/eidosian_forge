import abc
from tensorflow.python.util.tf_export import tf_export
@abc.abstractproperty
def parse_example_spec(self):
    """Returns a `tf.Example` parsing spec as dict.

    It is used for get_parsing_spec for `tf.io.parse_example`. Returned spec is
    a dict from keys ('string') to `VarLenFeature`, `FixedLenFeature`, and other
    supported objects. Please check documentation of `tf.io.parse_example` for
    all supported spec objects.

    Let's say a Feature column depends on raw feature ('raw') and another
    `FeatureColumn` (input_fc). One possible implementation of
    parse_example_spec is as follows:

    ```python
    spec = {'raw': tf.io.FixedLenFeature(...)}
    spec.update(input_fc.parse_example_spec)
    return spec
    ```
    """
    pass