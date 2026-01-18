from collections import namedtuple
from functools import singledispatch
import pennylane as qml
from pennylane import ops
from pennylane.measurements import MidMeasureMP
from .mpldrawer import MPLDrawer
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
from .style import _set_style
def tape_mpl(tape, wire_order=None, show_all_wires=False, decimals=None, style=None, *, fig=None, **kwargs):
    """Produces a matplotlib graphic from a tape.

    Args:
        tape (QuantumTape): the operations and measurements to draw

    Keyword Args:
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
        style (str): visual style of plot. Valid strings are ``{'black_white', 'black_white_dark', 'sketch',
            'sketch_dark', 'solarized_light', 'solarized_dark', 'default'}``. If no style is specified, the
            global style set with :func:`~.use_style` will be used, and the initial default is 'black_white'.
            If you would like to use your environment's current rcParams, set `style` to "rcParams".
            Setting style does not modify matplotlib global plotting settings.
        fontsize (float or str): fontsize for text. Valid strings are
            ``{'xx-small', 'x-small', 'small', 'medium', large', 'x-large', 'xx-large'}``.
            Default is ``14``.
        wire_options (dict): matplotlib formatting options for the wire lines
        label_options (dict): matplotlib formatting options for the wire labels
        active_wire_notches (bool): whether or not to add notches indicating active wires.
            Defaults to ``True``.
        fig (None or matplotlib Figure): Matplotlib figure to plot onto. If None, then create a new figure.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes._axes.Axes: The key elements for matplotlib's object oriented interface.

    **Example:**

    .. code-block:: python

        ops = [
            qml.QFT(wires=(0,1,2,3)),
            qml.IsingXX(1.234, wires=(0,2)),
            qml.Toffoli(wires=(0,1,2)),
            qml.CSWAP(wires=(0,2,3)),
            qml.RX(1.2345, wires=0),
            qml.CRZ(1.2345, wires=(3,0))
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

        fig, ax = tape_mpl(tape)
        fig.show()

    .. figure:: ../../_static/tape_mpl/default.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    .. details::
        :title: Usage Details

    **Decimals:**

    The keyword ``decimals`` controls how many decimal points to include when labelling the operations.
    The default value ``None`` omits parameters for brevity.

    .. code-block:: python

        ops = [qml.RX(1.23456, wires=0), qml.Rot(1.2345,2.3456, 3.456, wires=0)]
        measurements = [qml.expval(qml.Z(0))]
        tape2 = qml.tape.QuantumTape(ops, measurements)

        fig, ax = tape_mpl(tape2, decimals=2)

    .. figure:: ../../_static/tape_mpl/decimals.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    **Wires:**

    The keywords ``wire_order`` and ``show_all_wires`` control the location of wires from top to bottom.

    .. code-block:: python

        fig, ax = tape_mpl(tape, wire_order=[3,2,1,0])

    .. figure:: ../../_static/tape_mpl/wire_order.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    If a wire is in ``wire_order``, but not in the ``tape``, it will be omitted by default.  Only by selecting
    ``show_all_wires=True`` will empty wires be diplayed.

    .. code-block:: python

        fig, ax = tape_mpl(tape, wire_order=["aux"], show_all_wires=True)

    .. figure:: ../../_static/tape_mpl/show_all_wires.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    **Integration with matplotlib:**

    This function returns matplotlib figure and axes objects. Using these objects,
    users can perform further customization of the graphic.

    .. code-block:: python

        fig, ax = tape_mpl(tape)
        fig.suptitle("My Circuit", fontsize="xx-large")

        options = {'facecolor': "white", 'edgecolor': "#f57e7e", "linewidth": 6, "zorder": -1}
        box1 = plt.Rectangle((-0.5, -0.5), width=3.0, height=4.0, **options)
        ax.add_patch(box1)

        ax.annotate("CSWAP", xy=(3, 2.5), xycoords='data', xytext=(3.8,1.5), textcoords='data',
                    arrowprops={'facecolor': 'black'}, fontsize=14)

    .. figure:: ../../_static/tape_mpl/postprocessing.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    **Formatting:**

    PennyLane has inbuilt styles for controlling the appearance of the circuit drawings.
    All available styles can be determined by evaluating ``qml.drawer.available_styles()``.
    Any available string can then be passed via the kwarg ``style`` to change the settings for
    that plot. This will not affect style settings for subsequent matplotlib plots.

    .. code-block:: python

        fig, ax = tape_mpl(tape, style='sketch')

    .. figure:: ../../_static/tape_mpl/sketch_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    You can also control the appearance with matplotlib's provided tools, see the
    `matplotlib docs <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ .
    For example, we can customize ``plt.rcParams``. To use a customized appearance based on matplotlib's
    ``plt.rcParams``, ``qml.drawer.tape_mpl`` must be run with ``style="rcParams"``:

    .. code-block:: python

        plt.rcParams['patch.facecolor'] = 'mistyrose'
        plt.rcParams['patch.edgecolor'] = 'maroon'
        plt.rcParams['text.color'] = 'maroon'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['patch.linewidth'] = 4
        plt.rcParams['patch.force_edgecolor'] = True
        plt.rcParams['lines.color'] = 'indigo'
        plt.rcParams['lines.linewidth'] = 5
        plt.rcParams['figure.facecolor'] = 'ghostwhite'

        fig, ax = tape_mpl(tape, style="rcParams")

    .. figure:: ../../_static/tape_mpl/rcparams.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    The wires and wire labels can be manually formatted by passing in dictionaries of
    keyword-value pairs of matplotlib options. ``wire_options`` accepts options for lines,
    and ``label_options`` accepts text options.

    .. code-block:: python

        fig, ax = tape_mpl(tape, wire_options={'color':'teal', 'linewidth': 5},
                    label_options={'size': 20})

    .. figure:: ../../_static/tape_mpl/wires_labels.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """
    restore_params = {}
    if (update_style := (has_mpl and style != 'rcParams')):
        restore_params = mpl.rcParams.copy()
        _set_style(style)
    try:
        return _tape_mpl(tape, wire_order=wire_order, show_all_wires=show_all_wires, decimals=decimals, fig=fig, **kwargs)
    finally:
        if update_style:
            restore_params['interactive'] = mpl.rcParams['interactive']
            mpl.rcParams.update(restore_params)