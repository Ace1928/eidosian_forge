import base64
from io import BytesIO
import panel as pn
import param
from param.parameterized import bothmethod
from ...core import HoloMap
from ...core.options import Store
from ..renderer import HTML_TAGS, MIME_TYPES, Renderer
from .callbacks import callbacks
from .util import clean_internal_figure_properties
@bothmethod
def get_plot_state(self_or_cls, obj, doc=None, renderer=None, **kwargs):
    """
        Given a HoloViews Viewable return a corresponding figure dictionary.
        Allows cleaning the dictionary of any internal properties that were added
        """
    fig_dict = super().get_plot_state(obj, renderer, **kwargs)
    config = fig_dict.get('config', {})
    clean_internal_figure_properties(fig_dict)
    fig_dict = go.Figure(fig_dict).to_dict()
    fig_dict['config'] = config
    fig_dict.get('layout', {}).pop('template', None)
    return fig_dict