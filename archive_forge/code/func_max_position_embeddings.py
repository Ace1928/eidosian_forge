from ....configuration_utils import PretrainedConfig
from ....utils import logging
@max_position_embeddings.setter
def max_position_embeddings(self, value):
    raise NotImplementedError(f'The model {self.model_type} is one of the few models that has no sequence length limit.')