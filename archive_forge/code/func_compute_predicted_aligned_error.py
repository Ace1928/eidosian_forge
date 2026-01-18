from typing import Dict, Optional, Tuple
import torch
def compute_predicted_aligned_error(logits: torch.Tensor, max_bin: int=31, no_bins: int=64, **kwargs) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(0, max_bin, steps=no_bins - 1, device=logits.device)
    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_aligned_error, max_predicted_aligned_error = _calculate_expected_aligned_error(alignment_confidence_breaks=boundaries, aligned_distance_error_probs=aligned_confidence_probs)
    return {'aligned_confidence_probs': aligned_confidence_probs, 'predicted_aligned_error': predicted_aligned_error, 'max_predicted_aligned_error': max_predicted_aligned_error}