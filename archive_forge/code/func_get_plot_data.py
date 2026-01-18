from __future__ import annotations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pymatgen.util.plotting import pretty_plot
def get_plot_data(self, electrode, term_zero=True):
    """
        Args:
            electrode: Electrode object
            term_zero: If True append zero voltage point at the end.

        Returns:
            Plot data in x, y.
        """
    x = []
    y = []
    cap = 0
    for sub_electrode in electrode.get_sub_electrodes(adjacent_only=True):
        if self.hide_negative and sub_electrode.get_average_voltage() < 0:
            continue
        if self.xaxis in {'capacity_grav', 'capacity'}:
            x.append(cap)
            cap += sub_electrode.get_capacity_grav()
            x.append(cap)
        elif self.xaxis == 'capacity_vol':
            x.append(cap)
            cap += sub_electrode.get_capacity_vol()
            x.append(cap)
        elif self.xaxis == 'x_form':
            x.extend((sub_electrode.x_charge, sub_electrode.x_discharge))
        elif self.xaxis == 'frac_x':
            x.extend((sub_electrode.voltage_pairs[0].frac_charge, sub_electrode.voltage_pairs[0].frac_discharge))
        else:
            raise NotImplementedError('x_axis must be capacity_grav/capacity_vol/x_form/frac_x')
        y.extend([sub_electrode.get_average_voltage()] * 2)
    if term_zero:
        x.append(x[-1])
        y.append(0)
    return (x, y)