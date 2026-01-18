import bisect
import math
import warnings
from fractions import Fraction
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps
from .utils import tqdm
def subset(self, indices: List[int]) -> 'VideoClips':
    video_paths = [self.video_paths[i] for i in indices]
    video_pts = [self.video_pts[i] for i in indices]
    video_fps = [self.video_fps[i] for i in indices]
    metadata = {'video_paths': video_paths, 'video_pts': video_pts, 'video_fps': video_fps}
    return type(self)(video_paths, clip_length_in_frames=self.num_frames, frames_between_clips=self.step, frame_rate=self.frame_rate, _precomputed_metadata=metadata, num_workers=self.num_workers, _video_width=self._video_width, _video_height=self._video_height, _video_min_dimension=self._video_min_dimension, _video_max_dimension=self._video_max_dimension, _audio_samples=self._audio_samples, _audio_channels=self._audio_channels, output_format=self.output_format)