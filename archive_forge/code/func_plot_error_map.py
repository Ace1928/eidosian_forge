import math
from typing import List
import numpy as np
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.transpiler.coupling import CouplingMap
from .exceptions import VisualizationError
@_optionals.HAS_MATPLOTLIB.require_in_call
@_optionals.HAS_SEABORN.require_in_call
def plot_error_map(backend, figsize=(15, 12), show_title=True, qubit_coordinates=None):
    """Plots the error map of a given backend.

    Args:
        backend (Backend): Given backend.
        figsize (tuple): Figure size in inches.
        show_title (bool): Show the title or not.
        qubit_coordinates (Sequence): An optional sequence input (list or array being the
            most common) of 2d coordinates for each qubit. The length of the
            sequence much mast the number of qubits on the backend. The sequence
            should be the planar coordinates in a 0-based square grid where each
            qubit is located.

    Returns:
        Figure: A matplotlib figure showing error map.

    Raises:
        VisualizationError: The backend does not provide gate errors for the 'sx' gate.
        MissingOptionalLibraryError: If matplotlib or seaborn is not installed.

    Example:
        .. plot::
           :include-source:

            from qiskit.visualization import plot_error_map
            from qiskit.providers.fake_provider import GenericBackendV2

            backend = GenericBackendV2(num_qubits=5)
            plot_error_map(backend)
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import gridspec, ticker
    import seaborn as sns
    from .utils import matplotlib_close_if_inline
    color_map = sns.cubehelix_palette(reverse=True, as_cmap=True)
    backend_version = _get_backend_interface_version(backend)
    if backend_version <= 1:
        backend_name = backend.name()
        num_qubits = backend.configuration().n_qubits
        cmap = backend.configuration().coupling_map
        props = backend.properties()
        props_dict = props.to_dict()
        single_gate_errors = [0] * num_qubits
        read_err = [0] * num_qubits
        cx_errors = []
        for gate in props_dict['gates']:
            if gate['gate'] == 'sx':
                _qubit = gate['qubits'][0]
                for param in gate['parameters']:
                    if param['name'] == 'gate_error':
                        single_gate_errors[_qubit] = param['value']
                        break
                else:
                    raise VisualizationError(f"Backend '{backend}' did not supply an error for the 'sx' gate.")
        if cmap:
            directed = False
            if num_qubits < 20:
                for edge in cmap:
                    if not [edge[1], edge[0]] in cmap:
                        directed = True
                        break
            for line in cmap:
                for item in props_dict['gates']:
                    if item['qubits'] == line:
                        cx_errors.append(item['parameters'][0]['value'])
                        break
        for qubit in range(num_qubits):
            try:
                read_err[qubit] = props.readout_error(qubit)
            except BackendPropertyError:
                pass
    else:
        backend_name = backend.name
        num_qubits = backend.num_qubits
        cmap = backend.coupling_map
        two_q_error_map = {}
        single_gate_errors = [0] * num_qubits
        read_err = [0] * num_qubits
        cx_errors = []
        for gate, prop_dict in backend.target.items():
            if prop_dict is None or None in prop_dict:
                continue
            for qargs, inst_props in prop_dict.items():
                if inst_props is None:
                    continue
                if gate == 'measure':
                    if inst_props.error is not None:
                        read_err[qargs[0]] = inst_props.error
                elif len(qargs) == 1:
                    if inst_props.error is not None:
                        single_gate_errors[qargs[0]] = max(single_gate_errors[qargs[0]], inst_props.error)
                elif len(qargs) == 2:
                    if inst_props.error is not None:
                        two_q_error_map[qargs] = max(two_q_error_map.get(qargs, 0), inst_props.error)
        if cmap:
            directed = False
            if num_qubits < 20:
                for edge in cmap:
                    if not [edge[1], edge[0]] in cmap:
                        directed = True
                        break
            for line in cmap.get_edges():
                err = two_q_error_map.get(tuple(line), 0)
                cx_errors.append(err)
    single_gate_errors = 100 * np.asarray(single_gate_errors)
    avg_1q_err = np.mean(single_gate_errors)
    single_norm = matplotlib.colors.Normalize(vmin=min(single_gate_errors), vmax=max(single_gate_errors))
    q_colors = [matplotlib.colors.to_hex(color_map(single_norm(err))) for err in single_gate_errors]
    directed = False
    line_colors = []
    if cmap:
        cx_errors = 100 * np.asarray(cx_errors)
        avg_cx_err = np.mean(cx_errors)
        cx_norm = matplotlib.colors.Normalize(vmin=min(cx_errors), vmax=max(cx_errors))
        line_colors = [matplotlib.colors.to_hex(color_map(cx_norm(err))) for err in cx_errors]
    read_err = 100 * np.asarray(read_err)
    avg_read_err = np.mean(read_err)
    max_read_err = np.max(read_err)
    fig = plt.figure(figsize=figsize)
    gridspec.GridSpec(nrows=2, ncols=3)
    grid_spec = gridspec.GridSpec(12, 12, height_ratios=[1] * 11 + [0.5], width_ratios=[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    left_ax = plt.subplot(grid_spec[2:10, :1])
    main_ax = plt.subplot(grid_spec[:11, 1:11])
    right_ax = plt.subplot(grid_spec[2:10, 11:])
    bleft_ax = plt.subplot(grid_spec[-1, :5])
    if cmap:
        bright_ax = plt.subplot(grid_spec[-1, 7:])
    qubit_size = 28
    if num_qubits <= 5:
        qubit_size = 20
    plot_gate_map(backend, qubit_color=q_colors, line_color=line_colors, qubit_size=qubit_size, line_width=5, plot_directed=directed, ax=main_ax, qubit_coordinates=qubit_coordinates)
    main_ax.axis('off')
    main_ax.set_aspect(1)
    if cmap:
        single_cb = matplotlib.colorbar.ColorbarBase(bleft_ax, cmap=color_map, norm=single_norm, orientation='horizontal')
        tick_locator = ticker.MaxNLocator(nbins=5)
        single_cb.locator = tick_locator
        single_cb.update_ticks()
        single_cb.update_ticks()
        bleft_ax.set_title(f'H error rate (%) [Avg. = {round(avg_1q_err, 3)}]')
    if cmap is None:
        bleft_ax.axis('off')
        bleft_ax.set_title(f'H error rate (%) = {round(avg_1q_err, 3)}')
    if cmap:
        cx_cb = matplotlib.colorbar.ColorbarBase(bright_ax, cmap=color_map, norm=cx_norm, orientation='horizontal')
        tick_locator = ticker.MaxNLocator(nbins=5)
        cx_cb.locator = tick_locator
        cx_cb.update_ticks()
        bright_ax.set_title(f'CNOT error rate (%) [Avg. = {round(avg_cx_err, 3)}]')
    if num_qubits < 10:
        num_left = num_qubits
        num_right = 0
    else:
        num_left = math.ceil(num_qubits / 2)
        num_right = num_qubits - num_left
    left_ax.barh(range(num_left), read_err[:num_left], align='center', color='#DDBBBA')
    left_ax.axvline(avg_read_err, linestyle='--', color='#212121')
    left_ax.set_yticks(range(num_left))
    left_ax.set_xticks([0, round(avg_read_err, 2), round(max_read_err, 2)])
    left_ax.set_yticklabels([str(kk) for kk in range(num_left)], fontsize=12)
    left_ax.invert_yaxis()
    left_ax.set_title('Readout Error (%)', fontsize=12)
    for spine in left_ax.spines.values():
        spine.set_visible(False)
    if num_right:
        right_ax.barh(range(num_left, num_qubits), read_err[num_left:], align='center', color='#DDBBBA')
        right_ax.axvline(avg_read_err, linestyle='--', color='#212121')
        right_ax.set_yticks(range(num_left, num_qubits))
        right_ax.set_xticks([0, round(avg_read_err, 2), round(max_read_err, 2)])
        right_ax.set_yticklabels([str(kk) for kk in range(num_left, num_qubits)], fontsize=12)
        right_ax.invert_yaxis()
        right_ax.invert_xaxis()
        right_ax.yaxis.set_label_position('right')
        right_ax.yaxis.tick_right()
        right_ax.set_title('Readout Error (%)', fontsize=12)
    else:
        right_ax.axis('off')
    for spine in right_ax.spines.values():
        spine.set_visible(False)
    if show_title:
        fig.suptitle(f'{backend_name} Error Map', fontsize=24, y=0.9)
    matplotlib_close_if_inline(fig)
    return fig