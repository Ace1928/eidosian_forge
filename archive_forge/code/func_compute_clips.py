import bisect
import math
import warnings
from fractions import Fraction
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps
from .utils import tqdm
def compute_clips(self, num_frames: int, step: int, frame_rate: Optional[int]=None) -> None:
    """
        Compute all consecutive sequences of clips from video_pts.
        Always returns clips of size `num_frames`, meaning that the
        last few frames in a video can potentially be dropped.

        Args:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
            frame_rate (int, optional): The frame rate
        """
    self.num_frames = num_frames
    self.step = step
    self.frame_rate = frame_rate
    self.clips = []
    self.resampling_idxs = []
    for video_pts, fps in zip(self.video_pts, self.video_fps):
        clips, idxs = self.compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate)
        self.clips.append(clips)
        self.resampling_idxs.append(idxs)
    clip_lengths = torch.as_tensor([len(v) for v in self.clips])
    self.cumulative_sizes = clip_lengths.cumsum(0).tolist()