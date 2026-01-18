from __future__ import absolute_import
from __future__ import division
import pythreejs
import os
import time
import warnings
import tempfile
import uuid
import base64
from io import BytesIO as StringIO
import six
import numpy as np
import PIL.Image
import matplotlib.style
import ipywidgets
import IPython
from IPython.display import display
import ipyvolume as ipv
import ipyvolume.embed
from ipyvolume import utils
from . import ui
def animation_control(object, sequence_length=None, add=True, interval=200):
    """Animate scatter, quiver or mesh by adding a slider and play button.

    :param object: :any:`Scatter` or :any:`Mesh` object (having an sequence_index property), or a list of these to
                   control multiple.
    :param sequence_length: If sequence_length is None we try try our best to figure out, in case we do it badly,
            you can tell us what it should be. Should be equal to the S in the shape of the numpy arrays as for instance
            documented in :any:`scatter` or :any:`plot_mesh`.
    :param add: if True, add the widgets to the container, else return a HBox with the slider and play button. Useful when you
            want to customise the layout of the widgets yourself.
    :param interval: interval in msec between each frame
    :return: If add is False, if returns the ipywidgets.HBox object containing the controls
    """
    if isinstance(object, (list, tuple)):
        objects = object
    else:
        objects = [object]
    del object
    if sequence_length is None:
        sequence_lengths = []
        for object in objects:
            sequence_lengths_previous = list(sequence_lengths)
            values = [getattr(object, name) for name in 'x y z aux vx vy vz'.split() if hasattr(object, name)]
            values = [k for k in values if k is not None]
            values.sort(key=lambda key: -len(key.shape))
            try:
                sequence_length = values[0].shape[0]
                if isinstance(object, ipv.Mesh):
                    if len(values[0].shape) >= 2:
                        sequence_lengths.append(sequence_length)
                else:
                    sequence_lengths.append(sequence_length)
            except IndexError:
                pass
            if hasattr(object, 'color'):
                color = object.color
                if color is not None:
                    shape = color.shape
                    if len(shape) == 3:
                        sequence_lengths.append(shape[0])
            if len(sequence_lengths) == len(sequence_lengths_previous):
                raise ValueError('no frame dimension found for object: {}'.format(object))
        sequence_length = max(sequence_lengths)
    fig = gcf()
    fig.animation = interval
    fig.animation_exponent = 1.0
    play = ipywidgets.Play(min=0, max=sequence_length - 1, interval=interval, value=0, step=1)
    slider = ipywidgets.IntSlider(min=0, max=play.max, step=1)
    ipywidgets.jslink((play, 'value'), (slider, 'value'))
    for object in objects:
        ipywidgets.jslink((slider, 'value'), (object, 'sequence_index'))
    control = ipywidgets.HBox([play, slider])
    if add:
        current.container.children = current.container.children + [control]
    else:
        return control