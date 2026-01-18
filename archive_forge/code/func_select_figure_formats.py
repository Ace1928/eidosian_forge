from io import BytesIO
from binascii import b2a_base64
from functools import partial
import warnings
from IPython.core.display import _pngxy
from IPython.utils.decorators import flag_calls
def select_figure_formats(shell, formats, **kwargs):
    """Select figure formats for the inline backend.

    Parameters
    ----------
    shell : InteractiveShell
        The main IPython instance.
    formats : str or set
        One or a set of figure formats to enable: 'png', 'retina', 'jpeg', 'svg', 'pdf'.
    **kwargs : any
        Extra keyword arguments to be passed to fig.canvas.print_figure.
    """
    import matplotlib
    from matplotlib.figure import Figure
    svg_formatter = shell.display_formatter.formatters['image/svg+xml']
    png_formatter = shell.display_formatter.formatters['image/png']
    jpg_formatter = shell.display_formatter.formatters['image/jpeg']
    pdf_formatter = shell.display_formatter.formatters['application/pdf']
    if isinstance(formats, str):
        formats = {formats}
    formats = set(formats)
    [f.pop(Figure, None) for f in shell.display_formatter.formatters.values()]
    mplbackend = matplotlib.get_backend().lower()
    if mplbackend == 'nbagg' or mplbackend == 'module://ipympl.backend_nbagg':
        formatter = shell.display_formatter.ipython_display_formatter
        formatter.for_type(Figure, _reshow_nbagg_figure)
    supported = {'png', 'png2x', 'retina', 'jpg', 'jpeg', 'svg', 'pdf'}
    bad = formats.difference(supported)
    if bad:
        bs = '%s' % ','.join([repr(f) for f in bad])
        gs = '%s' % ','.join([repr(f) for f in supported])
        raise ValueError('supported formats are: %s not %s' % (gs, bs))
    if 'png' in formats:
        png_formatter.for_type(Figure, partial(print_figure, fmt='png', base64=True, **kwargs))
    if 'retina' in formats or 'png2x' in formats:
        png_formatter.for_type(Figure, partial(retina_figure, base64=True, **kwargs))
    if 'jpg' in formats or 'jpeg' in formats:
        jpg_formatter.for_type(Figure, partial(print_figure, fmt='jpg', base64=True, **kwargs))
    if 'svg' in formats:
        svg_formatter.for_type(Figure, partial(print_figure, fmt='svg', **kwargs))
    if 'pdf' in formats:
        pdf_formatter.for_type(Figure, partial(print_figure, fmt='pdf', base64=True, **kwargs))