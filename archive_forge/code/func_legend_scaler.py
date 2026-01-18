import base64
import json
import math
import os
import re
import struct
import typing
import zlib
from typing import Any, Callable, Union
from jinja2 import Environment, PackageLoader
def legend_scaler(legend_values, max_labels=10.0):
    """
    Downsamples the number of legend values so that there isn't a collision
    of text on the legend colorbar (within reason). The colorbar seems to
    support ~10 entries as a maximum.

    """
    if len(legend_values) < max_labels:
        legend_ticks = legend_values
    else:
        spacer = int(math.ceil(len(legend_values) / max_labels))
        legend_ticks = []
        for i in legend_values[::spacer]:
            legend_ticks += [i]
            legend_ticks += [''] * (spacer - 1)
    return legend_ticks