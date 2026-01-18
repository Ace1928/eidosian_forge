from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
def key_images() -> Tuple[pg.Surface, Dict[str, pg.Rect]]:
    """Return a keyboard keys image strip and a mapping of image locations

    The return tuple is a pygame.Surface and a dictionary keyed by key name and valued by a pygame.Rect.

    This function encapsulates the constants relevant to the keyboard image
    file. There are five key types. One is the black key. The other four
    white keys are determined by the proximity of the black keys. The plain
    white key has no black key adjacent to it. A white-left and white-right
    key has a black key to the left or right of it respectively. A white-center
    key has a black key on both sides. A key may have up to six related
    images depending on the state of adjacent keys to its right.

    """
    my_dir = os.path.split(os.path.abspath(__file__))[0]
    strip_file = os.path.join(my_dir, 'data', 'midikeys.png')
    white_key_width = 42
    white_key_height = 160
    black_key_width = 22
    black_key_height = 94
    strip = pg.image.load(strip_file)
    names = ['black none', 'black self', 'white none', 'white self', 'white self-white', 'white-left none', 'white-left self', 'white-left black', 'white-left self-black', 'white-left self-white', 'white-left all', 'white-center none', 'white-center self', 'white-center black', 'white-center self-black', 'white-center self-white', 'white-center all', 'white-right none', 'white-right self', 'white-right self-white']
    rects = {}
    for i in range(2):
        rects[names[i]] = pg.Rect(i * white_key_width, 0, black_key_width, black_key_height)
    for i in range(2, len(names)):
        rects[names[i]] = pg.Rect(i * white_key_width, 0, white_key_width, white_key_height)
    return (strip, rects)