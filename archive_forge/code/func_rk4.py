import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def rk4(x0, y0, f):
    ds = 0.01
    stotal = 0
    xi = x0
    yi = y0
    xb, yb = self.blank_pos(xi, yi)
    xf_traj = []
    yf_traj = []
    while check(xi, yi):
        xf_traj.append(xi)
        yf_traj.append(yi)
        try:
            k1x, k1y = f(xi, yi)
            k2x, k2y = f(xi + 0.5 * ds * k1x, yi + 0.5 * ds * k1y)
            k3x, k3y = f(xi + 0.5 * ds * k2x, yi + 0.5 * ds * k2y)
            k4x, k4y = f(xi + ds * k3x, yi + ds * k3y)
        except IndexError:
            break
        xi += ds * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        yi += ds * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
        if not check(xi, yi):
            break
        stotal += ds
        new_xb, new_yb = self.blank_pos(xi, yi)
        if new_xb != xb or new_yb != yb:
            if self.blank[new_yb, new_xb] == 0:
                self.blank[new_yb, new_xb] = 1
                xb_changes.append(new_xb)
                yb_changes.append(new_yb)
                xb = new_xb
                yb = new_yb
            else:
                break
        if stotal > 2:
            break
    return (stotal, xf_traj, yf_traj)