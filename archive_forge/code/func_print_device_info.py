from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
def print_device_info():
    pygame.midi.init()
    _print_device_info()
    pygame.midi.quit()