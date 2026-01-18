from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
def make_merged_spec(self, dev):
    """Returns a new DeviceSpec which incorporates `dev`.

    When combining specs, `dev` will take precedence over the current spec.
    So for instance:
    ```
    first_spec = tf.DeviceSpec(job=0, device_type="CPU")
    second_spec = tf.DeviceSpec(device_type="GPU")
    combined_spec = first_spec.make_merged_spec(second_spec)
    ```

    is equivalent to:
    ```
    combined_spec = tf.DeviceSpec(job=0, device_type="GPU")
    ```

    Args:
      dev: a `DeviceSpec`

    Returns:
      A new `DeviceSpec` which combines `self` and `dev`
    """
    return self.__class__(*self._get_combined_properties(dev))