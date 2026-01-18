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
def plot_state_paulivec(state, title='', figsize=None, color=None, ax=None, *, filename=None):
    """Plot the Pauli-vector representation of a quantum state as bar graph.

    The Pauli-vector of a density matrix :math:`\\rho` is defined by the expectation of each
    possible tensor product of single-qubit Pauli operators (including the identity), that is

    .. math ::

        \\rho = \\frac{1}{2^n} \\sum_{\\sigma \\in \\{I, X, Y, Z\\}^{\\otimes n}}
               \\mathrm{Tr}(\\sigma \\rho) \\sigma.

    This function plots the coefficients :math:`\\mathrm{Tr}(\\sigma\\rho)` as bar graph.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list or str): Color of the coefficient value bars.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.

    Returns:
         :class:`matplotlib:matplotlib.figure.Figure` :
            The matplotlib.Figure of the visualization if the
            ``ax`` kwarg is not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. plot::
           :include-source:

           # You can set a color for all the bars.

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_state_paulivec

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = Statevector(qc)
           plot_state_paulivec(state, color='midnightblue', title="New PauliVec plot")

        .. plot::
           :include-source:

           # If you introduce a list with less colors than bars, the color of the bars will
           # alternate following the sequence from the list.

           import numpy as np
           from qiskit.quantum_info import DensityMatrix
           from qiskit import QuantumCircuit
           from qiskit.visualization import plot_state_paulivec

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.cz(0, 1)
           qc.ry(np.pi/3, 0)
           qc.rx(np.pi/5, 1)

           matrix = DensityMatrix(qc)
           plot_state_paulivec(matrix, color=['crimson', 'midnightblue', 'seagreen'])
    """
    from matplotlib import pyplot as plt
    labels, values = _paulivec_data(state)
    numelem = len(values)
    if figsize is None:
        figsize = (7, 5)
    if color is None:
        color = '#648fff'
    ind = np.arange(numelem)
    width = 0.5
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()
    ax.grid(zorder=0, linewidth=1, linestyle='--')
    ax.bar(ind, values, width, color=color, zorder=2)
    ax.axhline(linewidth=1, color='k')
    ax.set_ylabel('Coefficients', fontsize=14)
    ax.set_xticks(ind)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(labels, fontsize=14, rotation=70)
    ax.set_xlabel('Pauli', fontsize=14)
    ax.set_ylim([-1, 1])
    ax.set_facecolor('#eeeeee')
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    ax.set_title(title, fontsize=16)
    if return_fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)