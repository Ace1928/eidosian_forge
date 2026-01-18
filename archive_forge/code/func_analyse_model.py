import collections
from .utils import ExplicitEnum, is_torch_available, logging
def analyse_model(self):
    self.module_names = {m: name for name, m in self.model.named_modules()}