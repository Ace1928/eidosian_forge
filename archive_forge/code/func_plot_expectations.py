import enum
from typing import Any, List, Optional, TYPE_CHECKING, Union
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from cirq import circuits, ops, study, value
from cirq._compat import proper_repr
def plot_expectations(self, ax: Optional[plt.Axes]=None, **plot_kwargs: Any) -> plt.Axes:
    """Plots the expectation values of Pauli operators versus delay time.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes containing the plot.
        """
    show_plot = not ax
    if show_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    assert ax is not None
    ax.set_ylim(ymin=-2, ymax=2)
    ax.plot(self._expectation_pauli_x['delay_ns'], self._expectation_pauli_x['value'], 'bo-', label='<X>', **plot_kwargs)
    ax.plot(self._expectation_pauli_y['delay_ns'], self._expectation_pauli_y['value'], 'go-', label='<Y>', **plot_kwargs)
    ax.set_xlabel('Delay between initialization and measurement (nanoseconds)')
    ax.set_ylabel('Pauli Operator Expectation')
    ax.set_title('T2 Decay Pauli Expectations')
    ax.legend()
    if show_plot:
        fig.show()
    return ax