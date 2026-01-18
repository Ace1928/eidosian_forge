from io import BytesIO
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
def test_bbox_inches_tight_layout_notconstrained(tmp_path):
    fig, ax = plt.subplots()
    fig.savefig(tmp_path / 'foo.png', bbox_inches='tight', pad_inches='layout')