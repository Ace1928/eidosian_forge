import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants
def from_prediction(features: FeatureDict, result: ModelOutput, b_factors: Optional[np.ndarray]=None, chain_index: Optional[np.ndarray]=None, remark: Optional[str]=None, parents: Optional[Sequence[str]]=None, parents_chain_index: Optional[Sequence[int]]=None) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      chain_index: (Optional) Chain indices for multi-chain predictions
      remark: (Optional) Remark about the prediction
      parents: (Optional) List of template names
    Returns:
      A protein instance.
    """
    return Protein(aatype=features['aatype'], atom_positions=result['final_atom_positions'], atom_mask=result['final_atom_mask'], residue_index=features['residue_index'] + 1, b_factors=b_factors if b_factors is not None else np.zeros_like(result['final_atom_mask']), chain_index=chain_index, remark=remark, parents=parents, parents_chain_index=parents_chain_index)