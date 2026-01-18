from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines, patches
from cirq.qis.states import validate_density_matrix
def plot_density_matrix(matrix: np.ndarray, ax: Optional[plt.Axes]=None, *, show_text: bool=False, title: Optional[str]=None) -> plt.Axes:
    """Generates a plot for a given density matrix.

    1. Each entry of the density matrix, a complex number, is plotted as an
    Argand Diagram where the partially filled red circle represents the magnitude
    and the line represents the phase angle, going anti-clockwise from positive x - axis.
    2. The blue rectangles on the diagonal elements represent the probability
    of measuring the system in state $|i\rangle$.
    Rendering scheme is inspired from https://algassert.com/quirk

    Args:
        matrix: The density matrix to visualize
        show_text: If true, the density matrix values are also shown as text labels
        ax: The axes to plot on
        title: Title of the plot
    """
    plt.style.use('ggplot')
    _padding_around_plot = 0.001
    matrix = matrix.astype(np.complex128)
    num_qubits = int(np.log2(matrix.shape[0]))
    validate_density_matrix(matrix, qid_shape=(2 ** num_qubits,))
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0 - _padding_around_plot, 2 ** num_qubits + _padding_around_plot)
    ax.set_ylim(0 - _padding_around_plot, 2 ** num_qubits + _padding_around_plot)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            _plot_element_of_density_matrix(ax, i, j, np.abs(matrix[i][-j - 1]), np.angle(matrix[i][-j - 1]), show_rect=i == matrix.shape[1] - j - 1, show_text=show_text)
    ticks, labels = (np.arange(0.5, matrix.shape[0]), [f'{'0' * (num_qubits - len(f'{i:b}'))}{i:b}' for i in range(matrix.shape[0])])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(reversed(labels))
    ax.set_facecolor('#eeeeee')
    if title is not None:
        ax.set_title(title)
    return ax