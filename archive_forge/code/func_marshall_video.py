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
def marshall_video(coordinates: str, proto: VideoProto, data: MediaData, mimetype: str='video/mp4', start_time: int=0, subtitles: SubtitleData=None, end_time: int | None=None, loop: bool=False) -> None:
    """Marshalls a video proto, using url processors as needed.

    Parameters
    ----------
    coordinates : str
    proto : the proto to fill. Must have a string field called "data".
    data : str, bytes, BytesIO, numpy.ndarray, or file opened with
           io.open().
        Raw video data or a string with a URL pointing to the video
        to load. Includes support for YouTube URLs.
        If passing the raw data, this must include headers and any other
        bytes required in the actual file.
    mimetype : str
        The mime type for the video file. Defaults to 'video/mp4'.
        See https://tools.ietf.org/html/rfc4281 for more info.
    start_time : int
        The time from which this element should start playing. (default: 0)
    subtitles: str, dict, or io.BytesIO
        Optional subtitle data for the video, supporting several input types:
        * None (default): No subtitles.
        * A string: File path to a subtitle file in '.vtt' or '.srt' formats, or the raw content of subtitles conforming to these formats.
            If providing raw content, the string must adhere to the WebVTT or SRT format specifications.
        * A dictionary: Pairs of labels and file paths or raw subtitle content in '.vtt' or '.srt' formats.
            Enables multiple subtitle tracks. The label will be shown in the video player.
            Example: {'English': 'path/to/english.vtt', 'French': 'path/to/french.srt'}
        * io.BytesIO: A BytesIO stream that contains valid '.vtt' or '.srt' formatted subtitle data.
        When provided, subtitles are displayed by default. For multiple tracks, the first one is displayed by default.
        Not supported for YouTube videos.
    end_time: int
            The time at which this element should stop playing
    loop: bool
        Whether the video should loop playback.
    """
    if start_time < 0 or (end_time is not None and end_time <= start_time):
        raise StreamlitAPIException('Invalid start_time and end_time combination.')
    proto.start_time = start_time
    if end_time is not None:
        proto.end_time = end_time
    proto.loop = loop
    proto.type = VideoProto.Type.NATIVE
    if isinstance(data, str) and url_util.is_url(data, allowed_schemas=('http', 'https', 'data')):
        if (youtube_url := _reshape_youtube_url(data)):
            proto.url = youtube_url
            proto.type = VideoProto.Type.YOUTUBE_IFRAME
            if subtitles:
                raise StreamlitAPIException('Subtitles are not supported for YouTube videos.')
        else:
            proto.url = data
    else:
        _marshall_av_media(coordinates, proto, data, mimetype)
    if subtitles:
        subtitle_items: list[tuple[str, str | Path | bytes | io.BytesIO]] = []
        if isinstance(subtitles, (str, bytes, io.BytesIO, Path)):
            subtitle_items.append(('default', subtitles))
        elif isinstance(subtitles, dict):
            subtitle_items.extend(subtitles.items())
        else:
            raise StreamlitAPIException(f'Unsupported data type for subtitles: {type(subtitles)}. Only str (file paths) and dict are supported.')
        for label, subtitle_data in subtitle_items:
            sub = proto.subtitles.add()
            sub.label = label or ''
            subtitle_coordinates = f'{coordinates}[subtitle{label}]'
            try:
                sub.url = process_subtitle_data(subtitle_coordinates, subtitle_data, label)
            except (TypeError, ValueError) as original_err:
                raise StreamlitAPIException(f'Failed to process the provided subtitle: {label}') from original_err