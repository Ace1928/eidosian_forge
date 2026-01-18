from __future__ import annotations
import json
from enum import Enum
from typing import TYPE_CHECKING, Dict, Final, Literal, Mapping, Union
from typing_extensions import TypeAlias
from streamlit.elements.lib.column_types import ColumnConfig, ColumnType
from streamlit.elements.lib.dicttools import remove_none_values
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.type_util import DataFormat, is_colum_type_arrow_incompatible
def process_config_mapping(column_config: ColumnConfigMappingInput | None=None) -> ColumnConfigMapping:
    """Transforms a user-provided column config mapping into a valid column config mapping
    that can be used by the frontend.

    Parameters
    ----------
    column_config: dict or None
        The user-provided column config mapping.

    Returns
    -------
    dict
        The transformed column config mapping.
    """
    if column_config is None:
        return {}
    transformed_column_config: ColumnConfigMapping = {}
    for column, config in column_config.items():
        if config is None:
            transformed_column_config[column] = ColumnConfig(hidden=True)
        elif isinstance(config, str):
            transformed_column_config[column] = ColumnConfig(label=config)
        elif isinstance(config, dict):
            transformed_column_config[column] = config
        else:
            raise StreamlitAPIException(f'Invalid column config for column `{column}`. Expected `None`, `str` or `dict`, but got `{type(config)}`.')
    return transformed_column_config