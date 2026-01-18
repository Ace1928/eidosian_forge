import numpy as np
from moviepy.decorators import convert_to_seconds, use_clip_fps_by_default
from ..io.preview import imdisplay
from .interpolators import Trajectory
def gatherClicks(t):
    imdisplay(clip.get_frame(t), screen)
    objects_to_click = nobjects
    clicks = []
    while objects_to_click:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_BACKSLASH:
                    return 'return'
                elif event.key == pg.K_ESCAPE:
                    raise KeyboardInterrupt()
            elif event.type == pg.MOUSEBUTTONDOWN:
                x, y = pg.mouse.get_pos()
                clicks.append((x, y))
                objects_to_click -= 1
    return clicks