from itertools import product
import numpy as np
from .utils import format_nvec
def radial_box(coeffs, n_inputs, ax, show_freqs=True, colour_dict=None, show_fliers=True):
    """Plot a list of sets of Fourier coefficients on a radial plot as box plots.

    Produces a 2-panel plot in which the left panel represents the real parts of
    Fourier coefficients. This method accepts multiple sets of coefficients, and
    plots the distribution of each coefficient as a boxplot.

    Args:
        coeffs (list[array[complex]]): A list of sets of Fourier coefficients. The shape of the
            coefficient arrays should resemble that of the output of numpy/scipy's ``fftn`` function, or
            :func:`~.pennylane.fourier.coefficients`.
        n_inputs (int): Dimension of the transformed function.
        ax (array[matplotlib.axes.Axes]): Axes to plot on. For this function, subplots
            must specify ``subplot_kw={"polar":True}`` upon construction.
        show_freqs (bool): Whether or not to label the frequencies on
            the radial axis. Turn off for large plots.
        colour_dict (dict[str, str]): Specify a colour mapping for positive and negative
            real/imaginary components. If none specified, will default to:
            ``{"real" : "red", "imag" : "black"}``
        showfliers (bool): Whether or not to plot outlying "fliers" on the boxplots.
        merge_plots (bool): Whether to plot real/complex values on the same panel, or
            on separate panels. Default is to plot real/complex values on separate panels.

    Returns:
        array[matplotlib.axes.Axes]: The axes after plotting is complete.

    **Example**

    Suppose we have the following quantum function:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit_with_weights(w, x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.Rot(*w[0], wires=0)
            qml.Rot(*w[1], wires=1)
            qml.CNOT(wires=[1, 0])

            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[1, 0])

            return qml.expval(qml.Z(0))

    We would like to compute and plot the distribution of Fourier coefficients
    for many random values of the weights ``w``. First, we generate all the coefficients:

    .. code-block:: python

        from functools import partial

        coeffs = []

        n_inputs = 2
        degree = 2

        for _ in range(100):
            weights = np.random.normal(0, 1, size=(2, 3))
            c = coefficients(partial(circuit_with_weights, weights), n_inputs, degree)
            coeffs.append(c)

    We can now plot by setting up a pair of ``matplotlib`` axes and passing them
    to the plotting function. Note that the axes passed must use polar coordinates.

    .. code-block:: python

        import matplotlib.pyplot as plt
        from pennylane.fourier.visualize import radial_box

        fig, ax = plt.subplots(
            1, 2, sharex=True, sharey=True,
            subplot_kw={"polar": True},
            figsize=(15, 8)
        )

        radial_box(coeffs, 2, ax, show_freqs=True, show_fliers=False)

    .. image:: ../../_static/fourier_vis_radial_box.png
        :align: center
        :width: 800px
        :target: javascript:void(0);

    """
    coeffs = _validate_coefficients(coeffs, n_inputs, True)
    if ax.size != 2:
        raise ValueError('Matplotlib axis should consist of two subplots.')
    if ax[0].name != 'polar' or ax[1].name != 'polar':
        raise ValueError('Matplotlib axes for radial_box must be polar.')
    if colour_dict is None:
        colour_dict = {'real': 'red', 'imag': 'black'}
    N = coeffs[0].size
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles[-N // 2 + 1:], angles[:-N // 2 + 1]))[::-1]
    width = (angles[1] - angles[0]) / 2
    nvecs_formatted, data = _extract_data_and_labels(coeffs)
    for data_type, a in zip(['real', 'imag'], ax):
        data_colour = colour_dict[data_type]
        a.boxplot(data[data_type], positions=angles, widths=width, boxprops={'facecolor': to_rgb(data_colour) + (0.4,), 'color': data_colour, 'edgecolor': data_colour}, medianprops={'color': data_colour, 'linewidth': 1.5}, flierprops={'markeredgecolor': data_colour}, whiskerprops={'color': data_colour}, capprops={'color': data_colour}, patch_artist=True, showfliers=show_fliers)
        a.set_thetagrids(180 / np.pi * angles, labels=nvecs_formatted)
        a.set_theta_zero_location('N')
        a.set_rlabel_position(0)
    for a in ax:
        if show_freqs:
            for label, angle in zip(a.get_xticklabels(), angles):
                x, y = label.get_position()
                lab = a.text(x, y, label.get_text(), transform=label.get_transform(), ha=label.get_ha(), va=label.get_va(), fontsize=14, color='grey')
                if angle > np.pi:
                    lab.set_rotation(180 / np.pi * angle + 90)
                else:
                    lab.set_rotation(180 / np.pi * angle + 270)
            a.tick_params(pad=7 * n_inputs)
        a.set_xticklabels([])
    return ax