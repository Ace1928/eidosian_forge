import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches
@image_comparison(['connection_styles.png'], style='mpl20', remove_text=True)
def test_connection_styles():
    styles = mpatches.ConnectionStyle.get_styles()
    n = len(styles)
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, n)
    for i, stylename in enumerate(sorted(styles)):
        patch = mpatches.FancyArrowPatch((0.1, i), (0.8, i + 0.5), arrowstyle='->', connectionstyle=stylename, mutation_scale=25)
        ax.add_patch(patch)