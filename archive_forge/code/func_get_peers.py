from abc import ABC, abstractmethod
from math import log as mlog
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
def get_peers(self, rotate: bool=False) -> Tuple[List[int], List[int]]:
    """Returns the out and in-peers corresponding to 'self.rank'"""
    if rotate:
        self._rotate_group_indices()
    out_peers, in_peers = ([], [])
    for group_index in self._group_indices:
        out_peers.append(self.phone_book[self.rank][group_index].dest)
        for rank, peers in enumerate(self.phone_book):
            if rank == self.rank:
                continue
            if self.rank * self.nprocs_per_node == peers[group_index].dest:
                in_peers.append(rank)
    return (out_peers, in_peers)