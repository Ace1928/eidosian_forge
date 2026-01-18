from unittest import mock
import matplotlib.pyplot as plt
def get_ax():
    """Create a plot and return its axes."""
    fig, ax = plt.subplots(1, 1)
    ax.plot([0, 200], [0, 200])
    ax.set_aspect(1.0)
    ax.figure.canvas.draw()
    return ax