import sys, traceback
from ..Qt import QtWidgets, QtGui
def stackFromFrame(frame, lastFrame=None):
    """Return (text, stack_frame) for the entire stack ending at *frame*

    If *lastFrame* is given and present in the stack, then the stack is truncated 
    at that frame.
    """
    lines = traceback.format_stack(frame)
    frames = []
    while frame is not None:
        frames.insert(0, frame)
        frame = frame.f_back
    if lastFrame is not None and lastFrame in frames:
        frames = frames[:frames.index(lastFrame) + 1]
    return list(zip(lines[:len(frames)], frames))