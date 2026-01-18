import numpy as np
def mplfig_to_npimage(fig):
    """ Converts a matplotlib figure to a RGB frame after updating the canvas"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    l, b, w, h = canvas.figure.bbox.bounds
    w, h = (int(w), int(h))
    buf = canvas.tostring_rgb()
    image = np.frombuffer(buf, dtype=np.uint8)
    return image.reshape(h, w, 3)