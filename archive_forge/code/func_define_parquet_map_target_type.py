from typing import Optional
from absl import flags
def define_parquet_map_target_type(flag_values: flags.FlagValues) -> flags.FlagHolder[Optional[str]]:
    return flags.DEFINE_enum('parquet_map_target_type', None, ['ARRAY_OF_STRUCT'], 'Specifies the parquet map type. If it is equal to ARRAY_OF_STRUCT, then a map_field will be represented with a repeated struct (that has key and value fields).', flag_values=flag_values)