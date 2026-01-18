import codecs
import os
from typing import TYPE_CHECKING, Sequence, Type, Union
from wandb import util
from wandb.sdk.lib import runid
from ._private import MEDIA_TMP
from .base_types.media import Media, _numpy_arrays_to_lists
from .base_types.wb_value import WBValue
from .image import Image
@classmethod
def make_plot_media(cls: Type['Plotly'], val: Union['plotly.Figure', 'matplotlib.artist.Artist']) -> Union[Image, 'Plotly']:
    if util.is_matplotlib_typename(util.get_full_typename(val)):
        if util.matplotlib_contains_images(val):
            return Image(val)
        val = util.matplotlib_to_plotly(val)
    return cls(val)