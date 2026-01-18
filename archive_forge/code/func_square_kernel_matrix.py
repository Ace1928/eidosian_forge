from itertools import product
import pennylane as qml
def square_kernel_matrix(X, kernel, assume_normalized_kernel=False):
    """Computes the square matrix of pairwise kernel values for a given dataset.

    Args:
        X (list[datapoint]): List of datapoints
        kernel ((datapoint, datapoint) -> float): Kernel function that maps
            datapoints to kernel value.
        assume_normalized_kernel (bool, optional): Assume that the kernel is normalized, in
            which case the diagonal of the kernel matrix is set to 1, avoiding unnecessary
            computations.

    Returns:
        array[float]: The square matrix of kernel values.

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

    We can then compute the kernel matrix on a set of 4 (random) feature
    vectors ``X`` via

    >>> X = np.random.random((4, 2))
    >>> qml.kernels.square_kernel_matrix(X, kernel)
    tensor([[1.        , 0.9532702 , 0.96864001, 0.90932897],
            [0.9532702 , 1.        , 0.99727485, 0.95685561],
            [0.96864001, 0.99727485, 1.        , 0.96605621],
            [0.90932897, 0.95685561, 0.96605621, 1.        ]], requires_grad=True)
    """
    N = qml.math.shape(X)[0]
    if assume_normalized_kernel and N == 1:
        return qml.math.eye(1, like=qml.math.get_interface(X))
    matrix = [None] * N ** 2
    for i in range(N):
        for j in range(i + 1, N):
            matrix[N * i + j] = (kernel_value := kernel(X[i], X[j]))
            matrix[N * j + i] = kernel_value
    if assume_normalized_kernel:
        one = qml.math.ones_like(matrix[1])
        for i in range(N):
            matrix[N * i + i] = one
    else:
        for i in range(N):
            matrix[N * i + i] = kernel(X[i], X[i])
    shape = (N, N) if qml.math.ndim(matrix[0]) == 0 else (N, N, qml.math.size(matrix[0]))
    return qml.math.moveaxis(qml.math.reshape(qml.math.stack(matrix), shape), -1, 0)