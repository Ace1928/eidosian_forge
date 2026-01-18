import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches
@image_comparison(['boxarrow_test_image.png'])
def test_boxarrow():
    styles = mpatches.BoxStyle.get_styles()
    n = len(styles)
    spacing = 1.2
    figheight = n * spacing + 0.5
    fig = plt.figure(figsize=(4 / 1.5, figheight / 1.5))
    fontsize = 0.3 * 72
    for i, stylename in enumerate(sorted(styles)):
        fig.text(0.5, ((n - i) * spacing - 0.5) / figheight, stylename, ha='center', size=fontsize, transform=fig.transFigure, bbox=dict(boxstyle=stylename, fc='w', ec='k'))