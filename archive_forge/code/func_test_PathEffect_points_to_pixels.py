import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
import matplotlib.patches as patches
def test_PathEffect_points_to_pixels():
    fig = plt.figure(dpi=150)
    p1, = plt.plot(range(10))
    p1.set_path_effects([path_effects.SimpleLineShadow(), path_effects.Normal()])
    renderer = fig.canvas.get_renderer()
    pe_renderer = path_effects.PathEffectRenderer(p1.get_path_effects(), renderer)
    assert renderer.points_to_pixels(15) == pe_renderer.points_to_pixels(15)