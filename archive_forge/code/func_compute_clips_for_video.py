import bisect
import math
import warnings
from fractions import Fraction
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union
import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps
from .utils import tqdm
@staticmethod
def compute_clips_for_video(video_pts: torch.Tensor, num_frames: int, step: int, fps: int, frame_rate: Optional[int]=None) -> Tuple[torch.Tensor, Union[List[slice], torch.Tensor]]:
    if fps is None:
        fps = 1
    if frame_rate is None:
        frame_rate = fps
    total_frames = len(video_pts) * (float(frame_rate) / fps)
    _idxs = VideoClips._resample_video_idx(int(math.floor(total_frames)), fps, frame_rate)
    video_pts = video_pts[_idxs]
    clips = unfold(video_pts, num_frames, step)
    if not clips.numel():
        warnings.warn("There aren't enough frames in the current video to get a clip for the given clip length and frames between clips. The video (and potentially others) will be skipped.")
    idxs: Union[List[slice], torch.Tensor]
    if isinstance(_idxs, slice):
        idxs = [_idxs] * len(clips)
    else:
        idxs = unfold(_idxs, num_frames, step)
    return (clips, idxs)