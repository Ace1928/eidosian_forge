import sys
from typing import List, Tuple
from typing import List
from functools import wraps
from time import time
import logging
def print_characters_in_colors(characters: List[str], colors: List[Tuple[int, int, int]]) -> None:
    """
    Prints each character in the provided list in a series of colors.

    :param characters: A list of characters to be printed.
    :param colors: A list of RGB color tuples.
    """
    for char in characters:
        print(f'Character: {char} - Unicode: {ord(char)}', end=' | ')
        for color in colors:
            print(f'\x1b[38;2;{color[0]};{color[1]};{color[2]}m{char}\x1b[0m', end=' ')
        print()