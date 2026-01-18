import inspect
import tensorflow.compat.v2 as tf
from keras.src.dtensor import dtensor_api as dtensor
def inject_mesh(init_method):
    """Inject DTensor mesh information to an object.

    This is useful for keras object like `Metric` and `Optimizer` which need
    DTensor mesh to create the weights, but doesn't want to change the current
    public API interface.

    This is for temporary usage and eventually the mesh/layout information will
    be public arguments in the `__init__` method.

    Sample usage:
    ```python
    class Accuracy(tf.keras.metrics.Metric):

      @inject_mesh
      def __init__(self, name='accuracy', dtype=None):
         super().__init__(**kwargs)

      acc = Accuracy(mesh=mesh)
      assert acc._mesh == mesh
    ```

    Args:
      init_method: the `__init__` method of the Keras class to annotate.

    Returns:
      the annotated __init__ method.
    """

    def _wrap_function(instance, *args, **kwargs):
        mesh = kwargs.pop('mesh', None)
        if mesh is not None:
            instance._mesh = mesh
        init_method(instance, *args, **kwargs)
    return tf.__internal__.decorator.make_decorator(target=init_method, decorator_func=_wrap_function)