import os
import json
import numpy as np
import ipywidgets as widgets
import pythreejs
import ipywebrtc
from IPython.display import display
def sync_camera(self):
    with self.output:
        index = self.select_keyframes.index
        if index is not None:
            self.camera.position = self.positions[index]
            self.camera.quaternion = self.quaternions[index]