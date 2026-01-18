import os
from typing import Callable, Optional
import gym
from gym import logger
from gym.wrappers.monitoring import video_recorder
def start_video_recorder(self):
    """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
    self.close_video_recorder()
    video_name = f'{self.name_prefix}-step-{self.step_id}'
    if self.episode_trigger:
        video_name = f'{self.name_prefix}-episode-{self.episode_id}'
    base_path = os.path.join(self.video_folder, video_name)
    self.video_recorder = video_recorder.VideoRecorder(env=self.env, base_path=base_path, metadata={'step_id': self.step_id, 'episode_id': self.episode_id})
    self.video_recorder.capture_frame()
    self.recorded_frames = 1
    self.recording = True