from kivy.app import App
from kivy.uix.videoplayer import VideoPlayer
from kivy.clock import Clock
import os
import time
def check_position(self, *args):
    if self.player.position > 0.1:
        self.stop_player()