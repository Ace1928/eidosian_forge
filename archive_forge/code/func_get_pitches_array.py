import json
import os
import urllib.request
def get_pitches_array(min_pitch, max_pitch):
    return list(range(min_pitch, max_pitch + 1))