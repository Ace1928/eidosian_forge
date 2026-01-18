from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
@keras_export('keras.activations.mish')
def mish(x):
    """Mish activation function.

    It is defined as:

    `mish(x) = x * tanh(softplus(x))`

    where `softplus` is defined as:

    `softplus(x) = log(exp(x) + 1)`

    Args:
        x: Input tensor.

    Reference:

    - [Misra, 2019](https://arxiv.org/abs/1908.08681)
    """
    x = backend.convert_to_tensor(x)
    return Mish.static_call(x)