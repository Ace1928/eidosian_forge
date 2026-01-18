from __future__ import annotations
import io
import re
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Final, Union, cast
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import runtime, type_util, url_util
from streamlit.elements.lib.subtitle_utils import process_subtitle_data
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Audio_pb2 import Audio as AudioProto
from streamlit.proto.Video_pb2 import Video as VideoProto
from streamlit.runtime import caching
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.runtime_util import duration_to_seconds
def marshall_audio(coordinates: str, proto: AudioProto, data: MediaData, mimetype: str='audio/wav', start_time: int=0, sample_rate: int | None=None, end_time: int | None=None, loop: bool=False) -> None:
    """Marshalls an audio proto, using data and url processors as needed.

    Parameters
    ----------
    coordinates : str
    proto : The proto to fill. Must have a string field called "url".
    data : str, bytes, BytesIO, numpy.ndarray, or file opened with
            io.open()
        Raw audio data or a string with a URL pointing to the file to load.
        If passing the raw data, this must include headers and any other bytes
        required in the actual file.
    mimetype : str
        The mime type for the audio file. Defaults to "audio/wav".
        See https://tools.ietf.org/html/rfc4281 for more info.
    start_time : int
        The time from which this element should start playing. (default: 0)
    sample_rate: int or None
        Optional param to provide sample_rate in case of numpy array
    end_time: int
        The time at which this element should stop playing
    loop: bool
        Whether the audio should loop playback.
    """
    proto.start_time = start_time
    if end_time is not None:
        proto.end_time = end_time
    proto.loop = loop
    if isinstance(data, str) and url_util.is_url(data, allowed_schemas=('http', 'https', 'data')):
        proto.url = data
    else:
        data = _maybe_convert_to_wav_bytes(data, sample_rate)
        _marshall_av_media(coordinates, proto, data, mimetype)