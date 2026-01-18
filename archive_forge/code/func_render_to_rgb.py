import numpy as np
def render_to_rgb(figure):
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data: np.ndarray = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    if close:
        plt.close(figure)
    return image_chw