from typing import Union
import torch
def insert_point_guard(self, insert_point: Union[torch._C.Node, torch._C.Block]):
    return _InsertPoint(self, insert_point)