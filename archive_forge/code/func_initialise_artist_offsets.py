from __future__ import annotations
import typing
from copy import copy
import pandas as pd
from matplotlib.animation import ArtistAnimation
from .exceptions import PlotnineError
def initialise_artist_offsets(n: int):
    """
            Initilise artists_offsets arrays to zero

            Parameters
            ----------
            n : int
                Number of axes to initialise artists for.
                The artists for each axes are tracked separately.
            """
    for artist_type in artist_offsets:
        artist_offsets[artist_type] = [0] * n