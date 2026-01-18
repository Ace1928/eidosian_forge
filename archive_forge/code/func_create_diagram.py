from collections import deque
import os
import typing
from typing import (
from abc import ABC, abstractmethod
from enum import Enum
import string
import copy
import warnings
import re
import sys
from collections.abc import Iterable
import traceback
import types
from operator import itemgetter
from functools import wraps
from threading import RLock
from pathlib import Path
from .util import (
from .exceptions import *
from .actions import *
from .results import ParseResults, _ParseResultsWithOffset
from .unicode import pyparsing_unicode
def create_diagram(self, output_html: Union[TextIO, Path, str], vertical: int=3, show_results_names: bool=False, show_groups: bool=False, embed: bool=False, **kwargs) -> None:
    """
        Create a railroad diagram for the parser.

        Parameters:

        - ``output_html`` (str or file-like object) - output target for generated
          diagram HTML
        - ``vertical`` (int) - threshold for formatting multiple alternatives vertically
          instead of horizontally (default=3)
        - ``show_results_names`` - bool flag whether diagram should show annotations for
          defined results names
        - ``show_groups`` - bool flag whether groups should be highlighted with an unlabeled surrounding box
        - ``embed`` - bool flag whether generated HTML should omit <HEAD>, <BODY>, and <DOCTYPE> tags to embed
          the resulting HTML in an enclosing HTML source
        - ``head`` - str containing additional HTML to insert into the <HEAD> section of the generated code;
          can be used to insert custom CSS styling
        - ``body`` - str containing additional HTML to insert at the beginning of the <BODY> section of the
          generated code

        Additional diagram-formatting keyword arguments can also be included;
        see railroad.Diagram class.
        """
    try:
        from .diagram import to_railroad, railroad_to_html
    except ImportError as ie:
        raise Exception('must ``pip install pyparsing[diagrams]`` to generate parser railroad diagrams') from ie
    self.streamline()
    railroad = to_railroad(self, vertical=vertical, show_results_names=show_results_names, show_groups=show_groups, diagram_kwargs=kwargs)
    if isinstance(output_html, (str, Path)):
        with open(output_html, 'w', encoding='utf-8') as diag_file:
            diag_file.write(railroad_to_html(railroad, embed=embed, **kwargs))
    else:
        output_html.write(railroad_to_html(railroad, embed=embed, **kwargs))