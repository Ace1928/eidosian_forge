from typing import List, Union
from functools import reduce
import colorsys
import numpy as np
from qiskit import user_config
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check
from .array import _num_to_latex, array_to_latex
from .utils import matplotlib_close_if_inline
from .exceptions import VisualizationError
@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_hinton(state, title='', figsize=None, ax_real=None, ax_imag=None, *, filename=None):
    """Plot a hinton diagram for the density matrix of a quantum state.

    The hinton diagram represents the values of a matrix using
    squares, whose size indicate the magnitude of their corresponding value
    and their color, its sign. A white square means the value is positive and
    a black one means negative.

    Args:
        state (Statevector or DensityMatrix or ndarray): An N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        filename (str): file path to save image to.
        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.

    Returns:
        :class:`matplotlib:matplotlib.figure.Figure` :
            The matplotlib.Figure of the visualization if
            neither ax_real or ax_imag is set.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. plot::
           :include-source:

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import DensityMatrix
            from qiskit.visualization import plot_state_hinton

            qc = QuantumCircuit(2)
            qc.h([0, 1])
            qc.cz(0,1)
            qc.ry(np.pi/3 , 0)
            qc.rx(np.pi/5, 1)

            state = DensityMatrix(qc)
            plot_state_hinton(state, title="New Hinton Plot")

    """
    from matplotlib import pyplot as plt
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError('Input is not a multi-qubit quantum state.')
    max_weight = 2 ** np.ceil(np.log(np.abs(rho.data).max()) / np.log(2))
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)
    if figsize is None:
        figsize = (8, 5)
    if not ax_real and (not ax_imag):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        if ax_real:
            fig = ax_real.get_figure()
        else:
            fig = ax_imag.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    column_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2 ** num)][::-1]
    ly, lx = datareal.shape
    if ax1:
        ax1.patch.set_facecolor('gray')
        ax1.set_aspect('equal', 'box')
        ax1.xaxis.set_major_locator(plt.NullLocator())
        ax1.yaxis.set_major_locator(plt.NullLocator())
        for (x, y), w in np.ndenumerate(datareal):
            plot_x, plot_y = (y, lx - x - 1)
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([0.5 + plot_x - size / 2, 0.5 + plot_y - size / 2], size, size, facecolor=color, edgecolor=color)
            ax1.add_patch(rect)
        ax1.set_xticks(0.5 + np.arange(lx))
        ax1.set_yticks(0.5 + np.arange(ly))
        ax1.set_xlim([0, lx])
        ax1.set_ylim([0, ly])
        ax1.set_yticklabels(row_names, fontsize=14)
        ax1.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax1.set_title('Re[$\\rho$]', fontsize=14)
    if ax2:
        ax2.patch.set_facecolor('gray')
        ax2.set_aspect('equal', 'box')
        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.yaxis.set_major_locator(plt.NullLocator())
        for (x, y), w in np.ndenumerate(dataimag):
            plot_x, plot_y = (y, lx - x - 1)
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle([0.5 + plot_x - size / 2, 0.5 + plot_y - size / 2], size, size, facecolor=color, edgecolor=color)
            ax2.add_patch(rect)
        ax2.set_xticks(0.5 + np.arange(lx))
        ax2.set_yticks(0.5 + np.arange(ly))
        ax2.set_xlim([0, lx])
        ax2.set_ylim([0, ly])
        ax2.set_yticklabels(row_names, fontsize=14)
        ax2.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax2.set_title('Im[$\\rho$]', fontsize=14)
    fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16)
    if ax_real is None and ax_imag is None:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)