from rpy2 import robjects
from rpy2.robjects.lib import ggplot2, grdevices
from IPython import get_ipython  # type: ignore
from IPython.core.display import Image  # type: ignore
def image_png(gg, width=800, height=400):
    with grdevices.render_to_bytesio(grdevices.png, type='cairo-png', width=width, height=height, antialias='subpixel') as b:
        robjects.r('print')(gg)
    data = b.getvalue()
    ip_img = Image(data=data, format='png', embed=True)
    return ip_img