import torch
from typing import Any, Set, Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OrdDict
def generate_table_visualization(self, feature_filter: str='', module_fqn_filter: str=''):
    """
        Takes in optional filter values and prints out formatted tables of the information.

        The reason for the two tables printed out instead of one large one are that they handle different things:
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

        Example Use:
            >>> # xdoctest: +SKIP("undefined variables")
            >>> mod_report_visualizer.generate_table_visualization(
            ...     feature_filter = "per_channel_min",
            ...     module_fqn_filter = "block1"
            ... )
            >>> # prints out neatly formatted table with per_channel_min info
            >>> # for all modules in block 1 of the model
        """
    if not got_tabulate:
        print('Make sure to install tabulate and try again.')
        return None
    table_dict = self.generate_filtered_tables(feature_filter, module_fqn_filter)
    tensor_headers, tensor_table = table_dict[self.TABLE_TENSOR_KEY]
    channel_headers, channel_table = table_dict[self.TABLE_CHANNEL_KEY]
    table_str = ''
    if len(tensor_headers) > self.NUM_NON_FEATURE_TENSOR_HEADERS:
        table_str += 'Tensor Level Information \n'
        table_str += tabulate(tensor_table, headers=tensor_headers)
    if len(channel_headers) > self.NUM_NON_FEATURE_CHANNEL_HEADERS:
        table_str += '\n\n Channel Level Information \n'
        table_str += tabulate(channel_table, headers=channel_headers)
    if table_str == '':
        table_str = 'No data points to generate table with.'
    print(table_str)