from colorsys import hsv_to_rgb, hls_to_rgb
from .libmp import NoConvergence
from .libmp.backend import xrange
def phase_color_function(ctx, z):
    if ctx.isinf(z):
        return (1.0, 1.0, 1.0)
    if ctx.isnan(z):
        return (0.5, 0.5, 0.5)
    pi = 3.1415926535898
    w = float(ctx.arg(z)) / pi
    w = max(min(w, 1.0), -1.0)
    for i in range(1, len(blue_orange_colors)):
        if blue_orange_colors[i][0] > w:
            a, (ra, ga, ba) = blue_orange_colors[i - 1]
            b, (rb, gb, bb) = blue_orange_colors[i]
            s = (w - a) / (b - a)
            return (ra + (rb - ra) * s, ga + (gb - ga) * s, ba + (bb - ba) * s)