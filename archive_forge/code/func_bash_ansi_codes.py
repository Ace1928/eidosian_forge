import logging
from logging import Formatter
from typing import NamedTuple, Tuple, Union
def bash_ansi_codes():
    """Utility function to generate a bash command for all ANSI color-codes in 24-bit format using both foreground and background colors."""
    str = '\n        for r in {0..255}; do\n            for g in {0..255}; do\n                for b in {0..255}; do\n                    echo -e "\\e[38;2;${r};${g};${b}m"\'\\\\e[38;2;\'"${r}"m" FOREGROUND\\e[0m"\n                    echo -e "\\e[48;2;${r};${g};${b}m"\'\\\\e[48;2;\'"${r}"m" FOREGROUND\\e[0m"\n                done\n            done\n        done'
    return str