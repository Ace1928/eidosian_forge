from parlai.agents.transformer.transformer import TransformerRankerAgent
from .wizard_dict import WizardDictAgent
import numpy as np
import torch
@staticmethod
def dictionary_class():
    return WizardDictAgent