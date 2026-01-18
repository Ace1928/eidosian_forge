import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads, Surface, transform
import time
def smooth_scale_bench(data):
    transform.smoothscale(data, (128, 128))