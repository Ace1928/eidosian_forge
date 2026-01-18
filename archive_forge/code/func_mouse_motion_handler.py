import glob
import os
import os.path as osp
import sys
import re
import copy
import time
import math
import logging
import itertools
from ast import literal_eval
from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter
import importlib
import memory_profiler as mp
def mouse_motion_handler(event):
    x, y = (event.xdata, event.ydata)
    if x is not None and y is not None:
        for coord, (name, text, rect) in rectangles.items():
            x0, y0, x1, y1 = coord
            if x0 < x < x1 and y0 < y < y1:
                if FLAME_PLOTTER_VARS['hovered_rect'] == rect:
                    return
                if FLAME_PLOTTER_VARS['hovered_rect'] is not None:
                    FLAME_PLOTTER_VARS['hovered_rect'].set_alpha(FLAME_PLOTTER_VARS['alpha'])
                    FLAME_PLOTTER_VARS['hovered_text'].set_color((0, 0, 0, 0))
                    FLAME_PLOTTER_VARS['hovered_rect'].set_linewidth(1)
                FLAME_PLOTTER_VARS['hovered_text'] = text
                FLAME_PLOTTER_VARS['hovered_rect'] = rect
                FLAME_PLOTTER_VARS['alpha'] = rect.get_alpha()
                FLAME_PLOTTER_VARS['hovered_rect'].set_alpha(0.8)
                FLAME_PLOTTER_VARS['hovered_rect'].set_linewidth(3)
                FLAME_PLOTTER_VARS['hovered_text'].set_color((0, 0, 0, 1))
                pl.draw()
                return
    if FLAME_PLOTTER_VARS['hovered_rect'] is not None:
        FLAME_PLOTTER_VARS['hovered_text'].set_color((0, 0, 0, 0))
        FLAME_PLOTTER_VARS['hovered_rect'].set_alpha(FLAME_PLOTTER_VARS['alpha'])
        FLAME_PLOTTER_VARS['hovered_rect'].set_linewidth(1)
        pl.draw()
        FLAME_PLOTTER_VARS['hovered_rect'] = None
        FLAME_PLOTTER_VARS['hovered_text'] = None