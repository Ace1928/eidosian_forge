from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from torch.nn import Module
from . import aligner, utils
def get_aligner(self) -> Aligner:
    """Instantiate an Aligner.

        Returns:
            Aligner
        """
    return aligner.Aligner(blank=0)