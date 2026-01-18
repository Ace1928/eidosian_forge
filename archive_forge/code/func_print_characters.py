import sys
from typing import List, Tuple
from typing import List
from functools import wraps
from time import time
import logging
def print_characters(self) -> None:
    """
        Prints each character in the list of characters in a series of colors.
        """
    for char in self.characters:
        print(f'Character: {char} - Unicode: {ord(char)}', end=' | ')
        for color in self.colors:
            print(f'\x1b[38;2;{color[0]};{color[1]};{color[2]}m{char}\x1b[0m', end=' ')
        print()