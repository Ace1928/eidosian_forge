import numpy as np
from ..math import frobenius_inner_product
from .utils import square_kernel_matrix
def polarity(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=True, normalize=False):
    """Polarity of a given kernel function.

    For a dataset with feature vectors :math:`\\{x_i\\}` and associated labels :math:`\\{y_i\\}`,
    the polarity of the kernel function :math:`k` is given by

    .. math ::

        \\operatorname{P}(k) = \\sum_{i,j=1}^n y_i y_j k(x_i, x_j)

    If the dataset is unbalanced, that is if the numbers of datapoints in the
    two classes :math:`n_+` and :math:`n_-` differ,
    ``rescale_class_labels=True`` will apply a rescaling according to
    :math:`\\tilde{y}_i = \\frac{y_i}{n_{y_i}}`. This is activated by default
    and only results in a prefactor that depends on the size of the dataset
    for balanced datasets.

    The keyword argument ``assume_normalized_kernel`` is passed to
    :func:`~.kernels.square_kernel_matrix`, for the computation
    :func:`~.utils.frobenius_inner_product` is used.

    Args:
        X (list[datapoint]): List of datapoints.
        Y (list[float]): List of class labels of datapoints, assumed to be either -1 or 1.
        kernel ((datapoint, datapoint) -> float): Kernel function that maps datapoints to kernel value.
        assume_normalized_kernel (bool, optional): Assume that the kernel is normalized, i.e.
            the kernel evaluates to 1 when both arguments are the same datapoint.
        rescale_class_labels (bool, optional): Rescale the class labels. This is important to take
            care of unbalanced datasets.
        normalize (bool): If True, rescale the polarity to the target_alignment.

    Returns:
        float: The kernel polarity.

    **Example:**

    Consider a simple kernel function based on :class:`~.templates.embeddings.AngleEmbedding`:

    .. code-block :: python

        dev = qml.device('default.qubit', wires=2, shots=None)
        @qml.qnode(dev)
        def circuit(x1, x2):
            qml.templates.AngleEmbedding(x1, wires=dev.wires)
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=dev.wires)
            return qml.probs(wires=dev.wires)

        kernel = lambda x1, x2: circuit(x1, x2)[0]

    We can then compute the polarity on a set of 4 (random) feature
    vectors ``X`` with labels ``Y`` via

    >>> X = np.random.random((4, 2))
    >>> Y = np.array([-1, -1, 1, 1])
    >>> qml.kernels.polarity(X, Y, kernel)
    tensor(0.04361349, requires_grad=True)
    """
    K = square_kernel_matrix(X, kernel, assume_normalized_kernel=assume_normalized_kernel)
    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)
    T = np.outer(_Y, _Y)
    return frobenius_inner_product(K, T, normalize=normalize)