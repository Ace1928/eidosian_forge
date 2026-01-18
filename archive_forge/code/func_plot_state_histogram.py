from typing import cast, Optional, Sequence, SupportsFloat, Union
import collections
import numpy as np
import matplotlib.pyplot as plt
import cirq.study.result as result
def plot_state_histogram(data: Union['result.Result', collections.Counter, Sequence[SupportsFloat]], ax: Optional[plt.Axes]=None, *, tick_label: Optional[Sequence[str]]=None, xlabel: Optional[str]='qubit state', ylabel: Optional[str]='result count', title: Optional[str]='Result State Histogram') -> plt.Axes:
    """Plot the state histogram from either a single result with repetitions or
       a histogram computed using `result.histogram()` or a flattened histogram
       of measurement results computed using `get_state_histogram`.

    Args:
        data:   The histogram values to plot. Possible options are:
                `result.Result`: Histogram is computed using
                    `get_state_histogram` and all 2 ** num_qubits values are
                    plotted, including 0s.
                `collections.Counter`: Only (key, value) pairs present in
                    collection are plotted.
                `Sequence[SupportsFloat]`: Values in the input sequence are
                    plotted. i'th entry corresponds to height of the i'th
                    bar in histogram.
        ax:      The Axes to plot on. If not given, a new figure is created,
                 plotted on, and shown.
        tick_label: Tick labels for the histogram plot in case input is not
                    `collections.Counter`. By default, label for i'th entry
                     is |i>.
        xlabel:  Label for the x-axis.
        ylabel:  Label for the y-axis.
        title:   Title of the plot.

    Returns:
        The axis that was plotted on.
    """
    show_fig = not ax
    if not ax:
        fig, ax = plt.subplots(1, 1)
        ax = cast(plt.Axes, ax)
    if isinstance(data, result.Result):
        values = get_state_histogram(data)
    elif isinstance(data, collections.Counter):
        tick_label, values = zip(*sorted(data.items()))
    else:
        values = np.array(data)
    if tick_label is None:
        tick_label = [str(i) for i in range(len(values))]
    ax.bar(np.arange(len(values)), values, tick_label=tick_label)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if show_fig:
        fig.show()
    return ax