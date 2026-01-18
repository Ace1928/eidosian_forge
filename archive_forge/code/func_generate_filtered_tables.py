import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def generate_filtered_tables(self, feature_filter: str='', module_fqn_filter: str='') -> Dict[str, Tuple[List, List]]:
    """
        Takes in optional filter values and generates two tables with desired information.

        The generated tables are presented in both a list-of-lists format

        The reason for the two tables are that they handle different things:
        1.) the first table handles all tensor level information
        2.) the second table handles and displays all channel based information

        The reasoning for this is that having all the info in one table can make it ambiguous which collected
            statistics are global, and which are actually per-channel, so it's better to split it up into two
            tables. This also makes the information much easier to digest given the plethora of statistics collected

        Tensor table columns:
            idx  layer_fqn  feature_1   feature_2   feature_3   .... feature_n
            ----  ---------  ---------   ---------   ---------        ---------

        Per-Channel table columns:
            idx  layer_fqn  channel  feature_1   feature_2   feature_3   .... feature_n
            ----  ---------  -------  ---------   ---------   ---------        ---------

        Args:
            feature_filter (str, optional): Filters the features presented to only those that
                contain this filter substring
                Default = "", results in all the features being printed
            module_fqn_filter (str, optional): Only includes modules that contains this string
                Default = "", results in all the modules in the reports to be visible in the table

        Returns a dictionary with two keys:
            (Dict[str, Tuple[List, List]]) A dict containing two keys:
            "tensor_level_info", "channel_level_info"
                Each key maps to a tuple with:
                    A list of the headers of each table
                    A list of lists containing the table information row by row
                    The 0th index row will contain the headers of the columns
                    The rest of the rows will contain data

        Example Use:
            >>> # xdoctest: +SKIP("undefined variables")
            >>> mod_report_visualizer.generate_filtered_tables(
            ...     feature_filter = "per_channel_min",
            ...     module_fqn_filter = "block1"
            ... ) # generates table with per_channel_min info for all modules in block 1 of the model
        """
    filtered_data: OrderedDict[str, Any] = self._get_filtered_data(feature_filter, module_fqn_filter)
    tensor_features: Set[str] = set()
    channel_features: Set[str] = set()
    num_channels: int = 0
    for module_fqn in filtered_data:
        for feature_name in filtered_data[module_fqn]:
            feature_data = filtered_data[module_fqn][feature_name]
            is_tensor: bool = isinstance(feature_data, torch.Tensor)
            is_not_zero_dim: bool = is_tensor and len(feature_data.shape) != 0
            if is_not_zero_dim or isinstance(feature_data, list):
                channel_features.add(feature_name)
                num_channels = len(feature_data)
            else:
                tensor_features.add(feature_name)
    tensor_features_list: List[str] = sorted(tensor_features)
    channel_features_list: List[str] = sorted(channel_features)
    tensor_headers, tensor_table = self._generate_tensor_table(filtered_data, tensor_features_list)
    channel_headers, channel_table = self._generate_channels_table(filtered_data, channel_features_list, num_channels)
    table_dict = {self.TABLE_TENSOR_KEY: (tensor_headers, tensor_table), self.TABLE_CHANNEL_KEY: (channel_headers, channel_table)}
    return table_dict