from __future__ import annotations
import typing
from copy import copy
import pandas as pd
from matplotlib.animation import ArtistAnimation
from .exceptions import PlotnineError
def get_frame_artists(axs: list[Axes]) -> list[Artist]:
    """
            Artists shown in a given frame

            Parameters
            ----------
            axs : list[Axes]
                Matplotlib axes that have had artists added
                to them.
            """
    frame_artists = []
    for i, ax in enumerate(axs):
        for name in artist_offsets:
            start = artist_offsets[name][i]
            new_artists = getattr(ax, name)[start:]
            frame_artists.extend(new_artists)
            artist_offsets[name][i] += len(new_artists)
    return frame_artists