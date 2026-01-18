import numpy as np
def mitigate_depolarizing_noise(K, num_wires, method, use_entries=None):
    """Estimate depolarizing noise rate(s) using on the diagonal entries of a kernel
    matrix and mitigate the noise, assuming a global depolarizing noise model.

    Args:
        K (array[float]): Noisy kernel matrix.
        num_wires (int): Number of wires/qubits of the quantum embedding kernel.
        method (``'single'`` | ``'average'`` | ``'split_channel'``): Strategy for mitigation

            * ``'single'``: An alias for ``'average'`` with ``len(use_entries)=1``.
            * ``'average'``: Estimate a global noise rate based on the average of the diagonal
              entries in ``use_entries``, which need to be measured on the quantum computer.
            * ``'split_channel'``: Estimate individual noise rates per embedding, requiring
              all diagonal entries to be measured on the quantum computer.
        use_entries (array[int]): Diagonal entries to use if method in ``['single', 'average']``.
            If ``None``, defaults to ``[0]`` (``'single'``) or ``range(len(K))`` (``'average'``).

    Returns:
        array[float]: Mitigated kernel matrix.

    Reference:
        This method is introduced in Section V in
        `arXiv:2105.02276 <https://arxiv.org/abs/2105.02276>`_.

    **Example:**

    For an example usage of ``mitigate_depolarizing_noise`` please refer to the
    `PennyLane demo on the kernel module <https://github.com/PennyLaneAI/qml/tree/master/demonstrations/tutorial_kernel_based_training.py>`_ or `the postprocessing demo for arXiv:2105.02276 <https://github.com/thubregtsen/qhack/blob/master/paper/post_processing_demo.py>`_.
    """
    dim = 2 ** num_wires
    if method == 'single':
        if use_entries is None:
            use_entries = (0,)
        if K[use_entries[0], use_entries[0]] <= 1 / dim:
            raise ValueError('The single noise mitigation method cannot be applied as the single diagonal element specified is too small.')
        diagonal_element = K[use_entries[0], use_entries[0]]
        noise_rate = (1 - diagonal_element) * dim / (dim - 1)
        mitigated_matrix = (K - noise_rate / dim) / (1 - noise_rate)
    elif method == 'average':
        if use_entries is None:
            diagonal_elements = np.diag(K)
        else:
            diagonal_elements = np.diag(K)[np.array(use_entries)]
        if np.mean(diagonal_elements) <= 1 / dim:
            raise ValueError('The average noise mitigation method cannot be applied as the average of the used diagonal terms is too small.')
        noise_rates = (1 - diagonal_elements) * dim / (dim - 1)
        mean_noise_rate = np.mean(noise_rates)
        mitigated_matrix = (K - mean_noise_rate / dim) / (1 - mean_noise_rate)
    elif method == 'split_channel':
        if np.any(np.diag(K) <= 1 / dim):
            raise ValueError('The split channel noise mitigation method cannot be applied to the input matrix as its diagonal terms are too small.')
        eff_noise_rates = np.clip((1 - np.diag(K)) * dim / (dim - 1), 0.0, 1.0)
        noise_rates = 1 - np.sqrt(1 - eff_noise_rates)
        inverse_noise = -np.outer(noise_rates, noise_rates) + noise_rates.reshape((1, len(K))) + noise_rates.reshape((len(K), 1))
        mitigated_matrix = (K - inverse_noise / dim) / (1 - inverse_noise)
    else:
        raise ValueError("Incorrect noise depolarization mitigation method specified. Accepted strategies are: 'single', 'average' and 'split_channel'.")
    return mitigated_matrix