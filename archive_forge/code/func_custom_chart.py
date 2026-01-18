from typing import Any, Dict, Optional, Tuple
from wandb.data_types import Table
from wandb.errors import Error
def custom_chart(vega_spec_name: str, data_table: Table, fields: Dict[str, Any], string_fields: Optional[Dict[str, Any]]=None, split_table: Optional[bool]=False) -> CustomChart:
    if string_fields is None:
        string_fields = {}
    if not isinstance(data_table, Table):
        raise Error(f'Expected `data_table` to be `wandb.Table` type, instead got {type(data_table).__name__}')
    return CustomChart(id=vega_spec_name, data=data_table, fields=fields, string_fields=string_fields, split_table=split_table)