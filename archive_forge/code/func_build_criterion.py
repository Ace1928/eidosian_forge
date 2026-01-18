from parlai.core.torch_agent import TorchAgent, Output
import torch
from parlai.core.agents import Agent
def build_criterion(self):
    return torch.nn.NLLLoss()