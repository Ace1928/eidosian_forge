import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def str_to_gesture(self, data):
    """Convert a unique string to a gesture."""
    io = BytesIO(zlib.decompress(base64.b64decode(data)))
    p = pickle.Unpickler(io)
    gesture = p.load()
    return gesture