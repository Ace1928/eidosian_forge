from __future__ import annotations
from typing import TYPE_CHECKING
def get_display_function(format: FigureFormat) -> Callable[[bytes], None]:
    """
    Return a function that will display the plot image
    """
    from IPython.display import SVG, Image, display_jpeg, display_pdf, display_png, display_svg

    def png(b: bytes):
        display_png(Image(b, format='png'))

    def retina(b: bytes):
        display_png(Image(b, format='png', retina=True))

    def jpeg(b: bytes):
        display_jpeg(Image(b, format='jpeg'))

    def svg(b: bytes):
        display_svg(SVG(b))

    def pdf(b: bytes):
        display_pdf(b, raw=True)
    lookup = {'png': png, 'retina': retina, 'jpeg': jpeg, 'jpg': jpeg, 'svg': svg, 'pdf': pdf}
    return lookup[format]