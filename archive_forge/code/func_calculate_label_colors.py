import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
@staticmethod
def calculate_label_colors(annotations: AnnotationList) -> Dict[str, str]:
    """
        Generates a color palette for all the labels in a given set of annotations

        Args:
          annotations (:obj:`Annotation`):
            A list of annotations

        Returns:
            :obj:`dict`: A dictionary mapping labels to colors in HSL format
        """
    if len(annotations) == 0:
        return {}
    labels = set(map(lambda x: x.label, annotations))
    num_labels = len(labels)
    h_step = int(255 / num_labels)
    if h_step < 20:
        h_step = 20
    s = 32
    l = 64
    h = 10
    colors = {}
    for label in sorted(labels):
        colors[label] = f'hsl({h},{s}%,{l}%'
        h += h_step
    return colors