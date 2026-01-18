import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def rk4_integrate(self, x0, y0):
    """
        RK4 forward and back trajectories from the initial conditions.

        Adapted from Bokeh's streamline -uses Runge-Kutta method to fill
        x and y trajectories then checks length of traj (s in units of axes)
        """

    def f(xi, yi):
        dt_ds = 1.0 / self.value_at(self.speed, xi, yi)
        ui = self.value_at(self.u, xi, yi)
        vi = self.value_at(self.v, xi, yi)
        return (ui * dt_ds, vi * dt_ds)

    def g(xi, yi):
        dt_ds = 1.0 / self.value_at(self.speed, xi, yi)
        ui = self.value_at(self.u, xi, yi)
        vi = self.value_at(self.v, xi, yi)
        return (-ui * dt_ds, -vi * dt_ds)
    check = lambda xi, yi: 0 <= xi < len(self.x) - 1 and 0 <= yi < len(self.y) - 1
    xb_changes = []
    yb_changes = []

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
    sf, xf_traj, yf_traj = rk4(x0, y0, f)
    sb, xb_traj, yb_traj = rk4(x0, y0, g)
    stotal = sf + sb
    x_traj = xb_traj[::-1] + xf_traj[1:]
    y_traj = yb_traj[::-1] + yf_traj[1:]
    if len(x_traj) < 1:
        return None
    if stotal > 0.2:
        initxb, inityb = self.blank_pos(x0, y0)
        self.blank[inityb, initxb] = 1
        return (x_traj, y_traj)
    else:
        for xb, yb in zip(xb_changes, yb_changes):
            self.blank[yb, xb] = 0
        return None