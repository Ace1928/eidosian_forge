from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
def apply_ops_2(self, dataframe):
    self.ctx['variables'][0].calculated_value = dataframe
    for op in self.ctx['operations']:
        op.execute()