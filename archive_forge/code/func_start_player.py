from kivy.app import App
from kivy.uix.videoplayer import VideoPlayer
from kivy.clock import Clock
import os
import time
def start_player(self, *args):
    self.player.state = 'play'
    self.start_t = time.perf_counter()