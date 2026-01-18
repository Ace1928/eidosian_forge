import io
import warnings
from typing import Any, Dict, Iterator, Optional
import torch
from ..utils import _log_api_usage_once
from ._video_opt import _HAS_VIDEO_OPT
def set_current_stream(self, stream: str) -> bool:
    """Set current stream.
        Explicitly define the stream we are operating on.

        Args:
            stream (string): descriptor of the required stream. Defaults to ``"video:0"``
                Currently available stream types include ``['video', 'audio']``.
                Each descriptor consists of two parts: stream type (e.g. 'video') and
                a unique stream id (which are determined by video encoding).
                In this way, if the video container contains multiple
                streams of the same type, users can access the one they want.
                If only stream type is passed, the decoder auto-detects first stream
                of that type and returns it.

        Returns:
            (bool): True on success, False otherwise
        """
    if self.backend == 'cuda':
        warnings.warn('GPU decoding only works with video stream.')
    if self.backend == 'pyav':
        stream_type = stream.split(':')[0]
        stream_id = 0 if len(stream.split(':')) == 1 else int(stream.split(':')[1])
        self.pyav_stream = {stream_type: stream_id}
        self._c = self.container.decode(**self.pyav_stream)
        return True
    return self._c.set_current_stream(stream)