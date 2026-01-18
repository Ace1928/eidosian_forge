from typing import Any, Dict, Optional
from ray.data.block import Block
from ray.util.annotations import PublicAPI
def get_filename_for_row(self, row: Dict[str, Any], task_index: int, block_index: int, row_index: int) -> str:
    file_id = f'{task_index:06}_{block_index:06}_{row_index:06}'
    return self._generate_filename(file_id)