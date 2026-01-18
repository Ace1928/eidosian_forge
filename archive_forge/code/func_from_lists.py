from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData
from vllm.utils import in_wsl, is_neuron
@classmethod
def from_lists(cls, temperatures: List[float], top_ps: List[float], top_ks: List[int], min_ps: List[float], presence_penalties: List[float], frequency_penalties: List[float], repetition_penalties: List[float], prompt_tokens: List[List[int]], output_tokens: List[List[int]], vocab_size: int, device: torch.device, dtype: torch.dtype) -> 'SamplingTensors':
    pin_memory = not in_wsl() and (not is_neuron())
    prompt_max_len = max((len(tokens) for tokens in prompt_tokens))
    prompt_padded_tokens = [tokens + [vocab_size] * (prompt_max_len - len(tokens)) for tokens in prompt_tokens]
    output_max_len = max((len(tokens) for tokens in output_tokens))
    output_padded_tokens = [tokens + [vocab_size] * (output_max_len - len(tokens)) for tokens in output_tokens]
    temperatures_t = torch.tensor(temperatures, device='cpu', dtype=dtype, pin_memory=pin_memory)
    top_ps_t = torch.tensor(top_ps, device='cpu', dtype=dtype, pin_memory=pin_memory)
    min_ps_t = torch.tensor(min_ps, device='cpu', dtype=dtype, pin_memory=pin_memory)
    presence_penalties_t = torch.tensor(presence_penalties, device='cpu', dtype=dtype, pin_memory=pin_memory)
    frequency_penalties_t = torch.tensor(frequency_penalties, device='cpu', dtype=dtype, pin_memory=pin_memory)
    repetition_penalties_t = torch.tensor(repetition_penalties, device='cpu', dtype=dtype, pin_memory=pin_memory)
    top_ks_t = torch.tensor(top_ks, device='cpu', dtype=torch.int, pin_memory=pin_memory)
    prompt_tensor = torch.tensor(prompt_padded_tokens, device='cpu', dtype=torch.long, pin_memory=pin_memory)
    output_tensor = torch.tensor(output_padded_tokens, device='cpu', dtype=torch.long, pin_memory=pin_memory)
    return cls(temperatures=temperatures_t.to(device=device, non_blocking=True), top_ps=top_ps_t.to(device=device, non_blocking=True), top_ks=top_ks_t.to(device=device, non_blocking=True), min_ps=min_ps_t.to(device=device, non_blocking=True), presence_penalties=presence_penalties_t.to(device=device, non_blocking=True), frequency_penalties=frequency_penalties_t.to(device=device, non_blocking=True), repetition_penalties=repetition_penalties_t.to(device=device, non_blocking=True), prompt_tokens=prompt_tensor.to(device=device, non_blocking=True), output_tokens=output_tensor.to(device=device, non_blocking=True))